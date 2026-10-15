import json

from nnworkbench.services.commands import prediction_command, training_command
from nnworkbench.services.device_probe import (
    DEVICE_CHOICES,
    DeviceInfo,
    default_choice,
    device_argument,
    device_hint,
    parse_probe,
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
