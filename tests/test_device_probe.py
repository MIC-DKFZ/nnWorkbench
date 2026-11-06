import json
import subprocess
import time

from nnworkbench.services import device_probe
from nnworkbench.services.commands import prediction_command, training_command
from nnworkbench.services.device_probe import (
    DEVICE_CHOICES,
    DeviceInfo,
    default_choice,
    device_argument,
    device_hint,
    parse_probe,
    probe_device,
)


def _output(**result) -> str:
    return "some warning\nNNWORKBENCH_DEVICE " + json.dumps(result)


def test_parse_probe_detects_cuda_mps_and_cpu() -> None:
    cuda = parse_probe(_output(torch="2.8.0", cuda_build="12.8", gpus=[["NVIDIA GeForce RTX 4090", 24 * 1024 ** 3]],
                               mps=False))
    assert (cuda.kind, cuda.summary, cuda.problem) == ("cuda", "NVIDIA GeForce RTX 4090 (24 GB)", "")
    assert parse_probe(_output(torch="2.8.0", cuda_build=None, gpus=[], mps=True)).kind == "mps"
    assert parse_probe(_output(torch="2.8.0", cuda_build=None, gpus=[], mps=False)).status == ("CPU only", "orange")


def test_parse_probe_reports_failures_as_unknown() -> None:
    assert parse_probe("Traceback ...").kind == "unknown"
    missing = parse_probe(_output(error="PyTorch could not be imported: No module named 'torch'"))
    assert missing.kind == "unknown" and "No module named" in missing.problem


def test_cpu_only_pytorch_with_nvidia_gpu_explains_the_reinstall() -> None:
    info = parse_probe(_output(torch="2.8.0+cpu", cuda_build=None, gpus=[], mps=False), nvidia_gpu=True)
    assert "without CUDA support" in info.problem and "download.pytorch.org/whl/cu" in info.fix_command
    assert "--force-reinstall" in info.fix_command  # PyTorch is already installed: pip would keep the CPU build


def test_each_device_choice_gives_the_same_command_everywhere() -> None:
    assert [device_argument(choice) for choice in DEVICE_CHOICES] == [None, "mps", "cpu"]


def test_detected_device_preselects_gpu_or_mps_but_never_cpu() -> None:
    assert default_choice(DeviceInfo("cuda")) == "GPU"
    assert default_choice(DeviceInfo("mps")) == "MPS"
    assert default_choice(DeviceInfo("cpu")) == "GPU"  # a copied command must not run on a cluster's CPU
    assert default_choice(None) == "GPU"


def test_device_hint_names_the_device_or_not_found() -> None:
    gpu = DeviceInfo("cuda", gpus=(("NVIDIA GeForce RTX 3090", 24.0),))
    assert device_hint("GPU", gpu) == "Device: NVIDIA GeForce RTX 3090 (24 GB)"
    assert device_hint("GPU", DeviceInfo("cpu")) == "Device: not found"
    assert device_hint("MPS", DeviceInfo("mps")) == "Device: Apple GPU"
    assert device_hint("MPS", gpu) == "Device: not found"
    assert device_hint("CPU", None) == "Device: CPU"
    assert device_hint("GPU", None) == "Device: checking…"


def test_commands_add_the_device_only_when_set() -> None:
    assert "-device" not in training_command(1, "nnUNetPlans", "2d", "T", "0")
    assert training_command(1, "nnUNetPlans", "2d", "T", "0", device="mps").endswith("-p nnUNetPlans -device mps")
    assert prediction_command(1, "/i", "/o", "P", "2d", device="cpu").endswith("-nps 3 -device cpu")


class _FakeProbe:
    """subprocess.run replaced: counts the probe processes and answers with the given output."""

    def __init__(self, monkeypatch, output: str) -> None:
        self.calls = 0
        self.output = output
        monkeypatch.setattr(subprocess, "run", self._run)

    def _run(self, command, **_kwargs):
        self.calls += 1
        return subprocess.CompletedProcess(command, 0, self.output, "")


def test_a_clean_result_is_cached_for_an_hour(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(device_probe.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    probe = _FakeProbe(monkeypatch, _output(torch="2.13.0", cuda_build="13.0", gpus=[["RTX 3090", 24 * 1024 ** 3]]))
    first = probe_device()
    assert first.kind == "cuda" and probe.calls == 1
    assert (tmp_path / "device_cache.json").is_file()  # cache_path points into tmp_path (conftest)
    assert probe_device() == first and probe.calls == 1  # answered from the cache, no process
    assert probe_device(use_cache=False) == first and probe.calls == 2  # asked to check again
    # an hour later the result is checked again
    stored = json.loads((tmp_path / "device_cache.json").read_text(encoding="utf-8"))
    stored["time"] = time.time() - device_probe.CACHE_SECONDS - 1
    (tmp_path / "device_cache.json").write_text(json.dumps(stored), encoding="utf-8")
    assert probe_device() == first and probe.calls == 3
    assert probe_device() == first and probe.calls == 3  # cached again


def test_the_cache_is_per_environment_computer_and_visible_gpus(monkeypatch) -> None:
    monkeypatch.setattr(device_probe.shutil, "which", lambda name: None)
    probe = _FakeProbe(monkeypatch, _output(torch="2.13.0", cuda_build=None, gpus=[]))
    assert probe_device().kind == "cpu" and probe.calls == 1
    monkeypatch.setattr(device_probe.platform, "node", lambda: "gpu-node-07")  # a cluster's other node
    assert probe_device().kind == "cpu" and probe.calls == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert probe.calls == 2 and probe_device().kind == "cpu" and probe.calls == 3
    monkeypatch.setattr(device_probe, "_torch_signature", lambda: "2.13.0 /env/torch/__init__.py 1700000000")
    assert probe_device().kind == "cpu" and probe.calls == 4  # torch reinstalled (also the same version, CUDA build)
    assert probe_device().kind == "cpu" and probe.calls == 4


def test_the_torch_signature_does_not_import_torch() -> None:
    import sys

    loaded = "torch" in sys.modules
    signature = device_probe._torch_signature()
    assert ("torch" in sys.modules) == loaded
    assert signature.split(" ")[0] == (device_probe.importlib.metadata.version("torch") if loaded or
                                       device_probe.importlib.util.find_spec("torch") else "")


def test_failures_and_problems_are_never_cached(monkeypatch, tmp_path) -> None:
    probe = _FakeProbe(monkeypatch, _output(error="PyTorch could not be imported: no module"))
    assert probe_device().kind == "unknown" and probe_device().kind == "unknown" and probe.calls == 2
    assert not (tmp_path / "device_cache.json").exists()
    # CPU-only PyTorch next to an NVIDIA GPU: the user will fix it and expects the next status to notice
    monkeypatch.setattr(device_probe.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    probe = _FakeProbe(monkeypatch, _output(torch="2.13.0+cpu", cuda_build=None, gpus=[]))
    assert probe_device().problem and probe_device().problem and probe.calls == 2
    assert not (tmp_path / "device_cache.json").exists()


def test_a_broken_or_unwritable_cache_is_ignored(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(device_probe.shutil, "which", lambda name: None)
    probe = _FakeProbe(monkeypatch, _output(torch="2.13.0", cuda_build=None, gpus=[], mps=True))
    (tmp_path / "device_cache.json").write_text("not json", encoding="utf-8")
    assert probe_device().kind == "mps" and probe.calls == 1
    (tmp_path / "device_cache.json").write_text(json.dumps({"key": {}, "time": "yesterday", "result": "x"}),
                                                encoding="utf-8")
    assert probe_device().kind == "mps" and probe.calls == 2
    (tmp_path / "blocked").write_text("a file, not a folder", encoding="utf-8")
    monkeypatch.setattr(device_probe, "cache_path", lambda: tmp_path / "blocked" / "device_cache.json")
    assert probe_device().kind == "mps" and probe_device().kind == "mps" and probe.calls == 4  # no cache, no error
