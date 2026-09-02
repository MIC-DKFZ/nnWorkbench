"""Run a command with nnWorkbench's saved nnU-Net environment."""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .path_setup import configured_paths
from .services.annotation_conversion import convert_annotations
from .services.dataset_scanner import discover
from .services.label_analysis import analyse_label_distribution


def run_with_configured_paths(command: Sequence[str]) -> int:
    """Run *command* with saved nnU-Net paths, returning its exit status."""
    if not command:
        raise ValueError("Specify a command to run.")
    environment = os.environ.copy()
    environment.update(configured_paths())
    return subprocess.run(command, env=environment, check=False).returncode


def raw_dataset_directory(dataset: str) -> Path:
    """Resolve a dataset ID or full directory name below the configured Raw root."""
    raw_root = Path(configured_paths()["nnUNet_raw"])
    datasets = discover(raw_root)
    if dataset.isdecimal() and int(dataset) in datasets:
        return datasets[int(dataset)][1]
    matches = [path for _name, path in datasets.values() if dataset.casefold() == path.name.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"No Raw dataset matching ID or directory name '{dataset}' was found in {raw_root}.")


def run_label_analysis(dataset: str, workers: int | None = None) -> int:
    """Analyse raw training labels and print progress for terminal users."""
    dataset_directory = raw_dataset_directory(dataset)
    analysis = analyse_label_distribution(
        dataset_directory, workers, progress=print,
    )
    print(f"Label analysis completed: {dataset_directory / 'class_distribution.json'} "
          f"({analysis['files_analyzed']} file(s))")
    return 0


def run_annotation_conversion(dataset: str, workers: int | None = None) -> int:
    """Convert annotation-workflow maps to labelsTr and print the outcome."""
    dataset_directory = raw_dataset_directory(dataset)
    result = convert_annotations(dataset_directory, workers, progress=print)
    print(f"Annotation conversion completed: {len(result.converted)} converted, "
          f"{len(result.warnings)} skipped or warned.")
    return 0


def _run_builtin_command(command: Sequence[str]) -> int | None:
    """Run an nnWorkbench data-management command, or return ``None`` for nnU-Net."""
    if not command:
        return None
    name, arguments = command[0], command[1:]
    if name == "label-analysis":
        parser = argparse.ArgumentParser(prog="nnworkbench-run label-analysis")
        parser.add_argument("dataset", help="Dataset ID or full DatasetNNN_Name directory name")
        parser.add_argument("--workers", type=int, default=None, help="Maximum worker processes")
        options = parser.parse_args(arguments)
        return run_label_analysis(options.dataset, options.workers)
    if name == "convert-annotations":
        parser = argparse.ArgumentParser(prog="nnworkbench-run convert-annotations")
        parser.add_argument("dataset", help="Dataset ID or full DatasetNNN_Name directory name")
        parser.add_argument("--workers", type=int, default=None, help="Maximum worker processes")
        options = parser.parse_args(arguments)
        return run_annotation_conversion(options.dataset, options.workers)
    return None


def main() -> None:
    """Run a command after applying the persisted nnWorkbench path configuration."""
    parser = argparse.ArgumentParser(
        description="Run nnU-Net commands or nnWorkbench dataset tools with saved paths.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="nnU-Net command or nnWorkbench subcommand")
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    try:
        builtin_exit_code = _run_builtin_command(command)
        raise SystemExit(builtin_exit_code if builtin_exit_code is not None else run_with_configured_paths(command))
    except (RuntimeError, TypeError, ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
