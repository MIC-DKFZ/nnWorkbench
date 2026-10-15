"""Run a command with nnWorkbench's saved nnU-Net environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .path_setup import configured_paths
from .services.annotation_conversion import convert_annotations
from .services.checkpoint_reduction import reduce_model_for_sharing
from .services.dataset_scanner import discover
from .services.image_stack import stack_image_folder
from .services.label_analysis import analyse_label_distribution


def find_executable(name: str) -> str:
    """Find a console script such as nnUNetv2_train: on PATH, or next to this Python (bin/ on Linux and macOS,
    Scripts\\ on Windows), so it is found even when the environment is not activated."""
    found = shutil.which(name)
    if found:
        return found
    folder = Path(sys.executable).parent
    for directory in (folder, folder / "Scripts", folder / "bin"):
        found = shutil.which(name, path=str(directory))
        if found:
            return found
    return name


def run_with_configured_paths(command: Sequence[str]) -> int:
    """Run *command* with saved nnU-Net paths, returning its exit status."""
    if not command:
        raise ValueError("Specify a command to run.")
    environment = os.environ.copy()
    environment.update(configured_paths())
    return subprocess.run([find_executable(command[0]), *command[1:]], env=environment, check=False).returncode


def raw_dataset_directory(dataset_id: int) -> Path:
    """Resolve a dataset ID below the configured Raw root."""
    raw_root = Path(configured_paths()["nnUNet_raw"])
    datasets = discover(raw_root)
    if dataset_id in datasets:
        return datasets[dataset_id][1]
    raise ValueError(f"No Raw dataset matching ID {dataset_id} was found in {raw_root}.")


def run_label_analysis(dataset_id: int, workers: int | None = None) -> int:
    """Analyse raw training labels and print progress for terminal users."""
    dataset_directory = raw_dataset_directory(dataset_id)
    analysis = analyse_label_distribution(
        dataset_directory, workers, progress=print,
    )
    print(f"Label analysis completed: {dataset_directory / 'class_distribution.json'} "
          f"({analysis['files_analyzed']} file(s))")
    return 0


def run_annotation_conversion(dataset_id: int, workers: int | None = None) -> int:
    """Convert annotation-workflow maps to labelsTr and print the outcome."""
    dataset_directory = raw_dataset_directory(dataset_id)
    result = convert_annotations(dataset_directory, workers, progress=print)
    print(f"Annotation conversion completed: {len(result.converted)} converted, "
          f"{len(result.warnings)} skipped or warned.")
    return 0


def run_image_stack(input_directory: Path, *, file_type: str, output_path: Path | None = None,
                    workers: int | None = None, scale_factor: int = 1,
                    spacing: tuple[float, float, float] | None = None, overwrite: bool = False,
                    show_mitk_output: bool = False) -> int:
    """Stack 2D image slices into one NIfTI volume and print concise progress."""
    stack_image_folder(
        input_directory,
        file_type=file_type,
        output_path=output_path,
        workers=workers,
        scale_factor=scale_factor,
        spacing=spacing,
        overwrite=overwrite,
        quiet=not show_mitk_output,
        progress=print,
    )
    return 0


def run_checkpoint_reduction(model_directory: Path, *, reset_trainer_name: bool = False) -> int:
    """Strip a trained model folder down to inference-only files and print progress."""
    folds = reduce_model_for_sharing(
        model_directory, reset_trainer_name=reset_trainer_name, progress=print,
    )
    print(f"Reduced {len(folds)} fold(s) in {model_directory} for sharing.")
    return 0


def run_dataset_inspection(dataset_id: int, index: int = 1) -> int:
    """Open a Raw dataset's images, labels and annotations in Napari, starting at the index-th image."""
    try:
        # Imported lazily: the lightweight remote installation has no Qt or Napari.
        from .services.napari_inspector import image_case_ids, open_case, related_files_for_case
    except ImportError as error:
        raise RuntimeError("Napari is not installed. Use the local installation to inspect datasets.") from error
    dataset_directory = raw_dataset_directory(dataset_id)
    cases = image_case_ids(dataset_directory)
    if not cases:
        raise ValueError(f"No images were found in {dataset_directory / 'imagesTr'}.")
    if not 1 <= index <= len(cases):
        raise ValueError(f"Image index must be between 1 and {len(cases)}.")
    image = related_files_for_case(dataset_directory, cases[index - 1]).images[0]
    print(f"Inspecting {image.name} ({index} of {len(cases)}) in {dataset_directory.name}")
    open_case(image)
    return 0


def _run_builtin_command(command: Sequence[str]) -> int | None:
    """Run an nnWorkbench data-management command, or return ``None`` for nnU-Net."""
    if not command:
        return None
    name, arguments = command[0], command[1:]
    if name == "label-analysis":
        parser = argparse.ArgumentParser(prog="nnworkbench-run label-analysis")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("--workers", type=int, default=None, help="Maximum worker processes")
        options = parser.parse_args(arguments)
        return run_label_analysis(options.dataset, options.workers)
    if name == "convert-annotations":
        parser = argparse.ArgumentParser(prog="nnworkbench-run convert-annotations")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("--workers", type=int, default=None, help="Maximum worker processes")
        options = parser.parse_args(arguments)
        return run_annotation_conversion(options.dataset, options.workers)
    if name == "inspect":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Open a dataset's images, labels and annotations in Napari.")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("-i", "--index", type=int, default=1,
                            help="Image to start with, counting from 1 in filename order (default: 1)")
        options = parser.parse_args(arguments)
        return run_dataset_inspection(options.dataset, options.index)
    if name == "stack-images":
        parser = argparse.ArgumentParser(prog=f"nnworkbench-run {name}")
        parser.add_argument("input_directory", type=Path,
                            help="Directory containing ordered 2D image slices")
        parser.add_argument("--file-type", required=True,
                            help="MITK-readable slice extension, for example .tif or .png")
        parser.add_argument("-o", "--output", type=Path,
                            help="Output image path (default: beside the input folder)")
        parser.add_argument("--workers", type=int, default=None,
                            help="Slice-loading worker threads (default: up to 4)")
        parser.add_argument("--scale-factor", type=int, default=1,
                            help="Integer downsampling factor for X, Y, and Z (default: 1)")
        parser.add_argument("--spacing", type=float, nargs=3, metavar=("X", "Y", "Z"),
                            help="Output voxel spacing in X Y Z order (default: 1 1 1)")
        parser.add_argument("--overwrite", action="store_true",
                            help="Replace an existing output file")
        parser.add_argument("--show-mitk-output", action="store_true",
                            help="Show MITK/ITK messages emitted while slices are loaded")
        options = parser.parse_args(arguments)
        return run_image_stack(
            options.input_directory,
            file_type=options.file_type,
            output_path=options.output,
            workers=options.workers,
            scale_factor=options.scale_factor,
            spacing=tuple(options.spacing) if options.spacing is not None else None,
            overwrite=options.overwrite,
            show_mitk_output=options.show_mitk_output,
        )
    if name == "reduce-checkpoints":
        parser = argparse.ArgumentParser(prog=f"nnworkbench-run {name}")
        parser.add_argument("model_directory", type=Path,
                            help="Trained model folder containing fold_* subdirectories")
        parser.add_argument("--reset-trainer-name", action="store_true",
                            help="Rewrite each checkpoint's trainer_name to nnUNetTrainer")
        options = parser.parse_args(arguments)
        return run_checkpoint_reduction(
            options.model_directory, reset_trainer_name=options.reset_trainer_name,
        )
    if name == "status":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Show what is done and what is next for a dataset: checks, preprocessing, models and folds "
                        "(the same states as the stage tabs in the GUI).")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("--json", action="store_true", help="Print the status as JSON, e.g. for scripts")
        options = parser.parse_args(arguments)
        from .status_report import run_status

        paths = configured_paths()
        roots = {key: Path(paths[f"nnUNet_{key}"]) if paths.get(f"nnUNet_{key}") else None
                 for key in ("raw", "preprocessed", "results")}
        return run_status(options.dataset, roots, as_json=options.json)
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
