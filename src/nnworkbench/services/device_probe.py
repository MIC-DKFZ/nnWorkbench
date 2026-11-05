"""Which device nnU-Net can use here (NVIDIA GPU, Apple GPU or CPU) and the matching -device argument.

PyTorch is imported in a separate Python process: importing it in the GUI would slow the start and set up CUDA
in the GUI process. That takes about two seconds, so a clean result is cached for an hour (``CACHE_SECONDS``) per
computer and environment: ``status`` and ``suggest``, which agents call again and again, answer at once.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .commands import WINDOWS

PYTORCH_URL = "https://pytorch.org/get-started/locally/"
CUDA_INSTALL = "pip install torch --index-url https://download.pytorch.org/whl/cu128"  # an example, see PYTORCH_URL

_PROBE = """
import json
result = {}
try:
    import torch
except Exception as error:
    result["error"] = f"PyTorch could not be imported: {error}"
else:
    result["torch"] = torch.__version__
    result["cuda_build"] = torch.version.cuda
    result["gpus"] = ([[torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory]
                       for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [])
    mps = getattr(torch.backends, "mps", None)
    result["mps"] = bool(mps is not None and mps.is_available())
print("NNWORKBENCH_DEVICE " + json.dumps(result))
"""
_RESULT_PREFIX = "NNWORKBENCH_DEVICE "  # the probe's result line (other output, e.g. warnings, is ignored)


@dataclass(frozen=True)
class DeviceInfo:
    """The probe result; ``kind`` is "cuda", "mps", "cpu" or "unknown" (probe failed)."""

    kind: str
    torch_version: str = ""
    cuda_build: str | None = None
    gpus: tuple[tuple[str, float], ...] = field(default=())  # name, memory in GB
    nvidia_gpu: bool = False  # nvidia-smi was found, i.e. an NVIDIA driver is installed
    error: str = ""

    @property
    def status(self) -> tuple[str, str]:
        """Short status and its tone for the Info dialog, e.g. ("GPU ready", "green")."""
        return {"cuda": ("GPU ready", "green"), "mps": ("Apple GPU ready", "green"), "cpu": ("CPU only", "orange"),
                }.get(self.kind, ("not detected", "gray"))

    @property
    def summary(self) -> str:
        """One line for hints, e.g. "NVIDIA GeForce RTX 4090 (24 GB)"."""
        if self.kind == "cuda":
            return ", ".join(f"{name} ({memory:.0f} GB)" for name, memory in self.gpus)
        return {"mps": "Apple GPU (MPS)", "cpu": "CPU only, no usable GPU",
                "unknown": "not detected"}.get(self.kind, self.kind)

    @property
    def problem(self) -> str:
        """Why no GPU is used and how to fix it (pointing to pytorch.org), or "" when there is nothing to fix."""
        if self.kind == "unknown":
            return f"{self.error or 'PyTorch could not be checked.'} Install PyTorch as described on pytorch.org."
        if self.kind == "cpu" and self.nvidia_gpu and not self.cuda_build:
            return (f"An NVIDIA GPU is installed, but PyTorch {self.torch_version} was installed without CUDA support "
                    "(the default on Windows). Reinstall PyTorch with CUDA as described on pytorch.org, e.g.:")
        if self.kind == "cpu" and self.nvidia_gpu:
            return (f"An NVIDIA GPU is installed, but PyTorch {self.torch_version} (CUDA {self.cuda_build}) cannot "
                    "use it. Check that the NVIDIA driver supports this CUDA version (nvidia-smi), or install a "
                    "matching PyTorch build as described on pytorch.org.")
        return ""

    @property
    def fix_command(self) -> str:
        """An example command that fixes the problem, or ""."""
        return CUDA_INSTALL if self.kind == "cpu" and self.nvidia_gpu and not self.cuda_build else ""


def parse_probe(output: str, nvidia_gpu: bool = False) -> DeviceInfo:
    """Turn the probe's output into a DeviceInfo (other output, e.g. warnings, is ignored)."""
    line = _result_line(output)
    if line is None:
        return DeviceInfo("unknown", nvidia_gpu=nvidia_gpu, error="The PyTorch check gave no result.")
    try:
        data = json.loads(line.split(" ", 1)[1])
    except json.JSONDecodeError:
        return DeviceInfo("unknown", nvidia_gpu=nvidia_gpu, error="The PyTorch check gave no result.")
    if data.get("error"):
        return DeviceInfo("unknown", nvidia_gpu=nvidia_gpu, error=str(data["error"]))
    gpus = tuple((str(name), int(memory) / 1024 ** 3) for name, memory in data.get("gpus", []))
    kind = "cuda" if gpus else "mps" if data.get("mps") else "cpu"
    return DeviceInfo(kind, str(data.get("torch", "")), data.get("cuda_build"), gpus, nvidia_gpu)


def _result_line(output: str) -> str | None:
    return next((line for line in reversed(output.splitlines()) if line.startswith(_RESULT_PREFIX)), None)


CACHE_SECONDS = 3600  # a cached result is used for this long


def cache_path() -> Path:
    """Where the last clean probe result is kept: next to the saved nnU-Net paths (per user)."""
    from nnworkbench.path_setup import configuration_path

    return configuration_path().parent / "device_cache.json"


def _cache_key(nvidia_smi: str | None) -> dict[str, str]:
    """What the result depends on: the environment (sys.prefix: the same for python and pythonw, a shortcut and a
    terminal) and its torch package, the NVIDIA driver, the computer (a cluster's login node and GPU nodes share
    the home folder) and the visible GPUs."""
    return {"environment": sys.prefix, "torch": _torch_signature(), "nvidia_smi": nvidia_smi or "",
            "host": platform.node(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def _torch_signature() -> str:
    """The installed torch package without importing it: its version from the package metadata, and where and when
    its files were installed (the metadata version is the same for the CPU and the CUDA build of a release)."""
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        version = ""
    try:
        origin = importlib.util.find_spec("torch").origin or ""
        installed = os.path.getmtime(origin)
    except (ImportError, ValueError, AttributeError, OSError):
        origin, installed = "", 0.0
    return f"{version} {origin} {installed:.0f}"


def _cached_result(key: dict[str, str]) -> str | None:
    """The cached probe output for *key* while it is fresh; None otherwise (also for an unreadable file)."""
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("key") != key or not isinstance(data.get("result"), str):
        return None
    written = data.get("time")
    if not isinstance(written, (int, float)) or isinstance(written, bool):
        return None
    age = time.time() - written
    return data["result"] if 0 <= age <= CACHE_SECONDS else None  # a time in the future: the clock changed


def _store_result(key: dict[str, str], result: str) -> None:
    """Keep the probe's result line; written atomically, as several commands may run at once. A folder that cannot
    be written (read-only configuration) only means the next call probes again."""
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".device.", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump({"key": key, "time": time.time(), "result": result}, file)
            os.replace(temporary, path)
        except Exception:
            os.unlink(temporary)
            raise
    except OSError:
        pass


def probe_device(timeout: float = 60, use_cache: bool = True) -> DeviceInfo:
    """Check which device PyTorch can use, in a separate Python process (takes a few seconds). With ``use_cache`` a
    clean result from the last hour is reused (see ``cache_path``); a failed check or one with a problem to fix is
    never cached, so the next call sees the fix."""
    nvidia_smi = shutil.which("nvidia-smi")
    nvidia_gpu = nvidia_smi is not None
    key = _cache_key(nvidia_smi)
    if use_cache:
        cached = _cached_result(key)
        if cached is not None:
            return parse_probe(cached, nvidia_gpu)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
            # no console window flashing up when the GUI was started without one (Windows)
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if WINDOWS else 0)
    except (OSError, subprocess.SubprocessError) as error:
        return DeviceInfo("unknown", nvidia_gpu=nvidia_gpu, error=f"The PyTorch check failed: {error}")
    info = parse_probe(completed.stdout, nvidia_gpu)
    if use_cache and info.kind != "unknown" and not info.problem:
        _store_result(key, _result_line(completed.stdout) or "")
    return info


# ------------------------------------------------------------ the Device choice in Train and Predict
# Each choice always gives the same command, on every system; the detected device only preselects one.

DEVICE_CHOICES = ("GPU", "MPS", "CPU")
_ARGUMENTS = {"GPU": None, "MPS": "mps", "CPU": "cpu"}  # None: nnU-Net's default (cuda), no -device


def device_argument(choice: str) -> str | None:
    """The value for -device, or None for an NVIDIA GPU (nnU-Net's default, so the command stays standard)."""
    return _ARGUMENTS[choice]


def default_choice(info: DeviceInfo | None) -> str:
    """The choice to preselect: MPS when the Apple GPU is the only GPU here, otherwise GPU.

    CPU is never preselected: a command copied to a cluster should not run on its CPU by accident.
    """
    return "MPS" if info is not None and info.kind == "mps" else "GPU"


def device_hint(choice: str, info: DeviceInfo | None) -> str:
    """The device the choice uses on this computer, for the hint below it in the Train and Predict dialogs."""
    if choice == "CPU":
        return "Device: CPU"
    if info is None:
        return "Device: checking…"
    if choice == "MPS":
        return "Device: Apple GPU" if info.kind == "mps" else "Device: not found"
    return f"Device: {info.summary}" if info.kind == "cuda" else "Device: not found"
