"""Run a command with nnWorkbench's saved nnU-Net environment."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Sequence

from .path_setup import configured_paths


def run_with_configured_paths(command: Sequence[str]) -> int:
    """Run *command* with saved nnU-Net paths, returning its exit status."""
    if not command:
        raise ValueError("Specify a command to run.")
    environment = os.environ.copy()
    environment.update(configured_paths())
    return subprocess.run(command, env=environment, check=False).returncode


def main() -> None:
    """Run a command after applying the persisted nnWorkbench path configuration."""
    parser = argparse.ArgumentParser(description="Run a command with saved nnWorkbench nnU-Net paths.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command and arguments to run")
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    try:
        raise SystemExit(run_with_configured_paths(command))
    except (RuntimeError, TypeError, ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
