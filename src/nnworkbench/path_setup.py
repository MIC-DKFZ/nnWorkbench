"""Persist and retrieve nnWorkbench's nnU-Net directory configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from shlex import quote


def configuration_path() -> Path:
    """Return the platform-appropriate user configuration file."""
    if sys.platform == "darwin":
        directory = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        directory = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        directory = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return directory / "nnworkbench" / "paths.json"


def trainer_directory() -> Path:
    """Return the installed directory supplied to ``nnUNet_extTrainer``."""
    return Path(__file__).parent / "nnunet_trainers"


def save_paths(raw: str, preprocessed: str, results: str) -> None:
    """Store paths in the user's nnWorkbench configuration."""
    path = configuration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {"raw": raw, "preprocessed": preprocessed, "results": results}
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def configured_paths() -> dict[str, str]:
    """Load the configured paths and bundled trainer extension directory."""
    try:
        values = json.loads(configuration_path().read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError("No nnWorkbench paths are configured. Run nnworkbench-set-paths first.") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid nnWorkbench path configuration: {configuration_path()}") from error
    if not isinstance(values, dict):
        raise TypeError(f"Invalid nnWorkbench path configuration: {configuration_path()}")
    return {
        "nnUNet_raw": str(values.get("raw", "")),
        "nnUNet_preprocessed": str(values.get("preprocessed", "")),
        "nnUNet_results": str(values.get("results", "")),
        "nnUNet_extTrainer": str(trainer_directory()),
    }


def paths_report() -> str:
    """Return the configured paths in a copyable shell-assignment format."""
    return "\n".join(f"{name}={quote(value)}" for name, value in configured_paths().items())


def set_paths_command(raw: str, preprocessed: str, results: str) -> str:
    """Return a pasteable command that persists these paths on a remote system."""
    arguments = " ".join((
        f"--raw {quote(raw)}",
        f"--preprocessed {quote(preprocessed)}",
        f"--results {quote(results)}",
    ))
    return f"nnworkbench-set-paths {arguments}"


def set_paths_main() -> None:
    """Save the supplied nnU-Net directories."""
    parser = argparse.ArgumentParser(description="Save nnWorkbench's nnU-Net directory paths.")
    parser.add_argument("--raw", required=True, help="Path to nnUNet_raw")
    parser.add_argument("--preprocessed", required=True, help="Path to nnUNet_preprocessed")
    parser.add_argument("--results", required=True, help="Path to nnUNet_results")
    arguments = parser.parse_args()
    save_paths(arguments.raw, arguments.preprocessed, arguments.results)
    print(f"Saved nnWorkbench paths to {configuration_path()}")


def get_paths_main() -> None:
    """Print paths saved by :func:`set_paths_main`."""
    try:
        print(paths_report())
    except (RuntimeError, TypeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    set_paths_main()
