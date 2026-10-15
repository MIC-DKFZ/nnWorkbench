"""Which device nnU-Net can use here (NVIDIA GPU, Apple GPU or CPU) and the matching -device argument.

PyTorch is imported in a separate Python process: importing it in the GUI would slow the start and set up CUDA
in the GUI process.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

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
    line = next((line for line in reversed(output.splitlines()) if line.startswith("NNWORKBENCH_DEVICE ")), None)
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


def probe_device(timeout: float = 60) -> DeviceInfo:
    """Check in a separate Python process which device PyTorch can use (takes a few seconds)."""
    nvidia_gpu = shutil.which("nvidia-smi") is not None
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
            # no console window flashing up when the GUI was started without one (Windows)
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if WINDOWS else 0)
    except (OSError, subprocess.SubprocessError) as error:
        return DeviceInfo("unknown", nvidia_gpu=nvidia_gpu, error=f"The PyTorch check failed: {error}")
    return parse_probe(completed.stdout, nvidia_gpu)


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
