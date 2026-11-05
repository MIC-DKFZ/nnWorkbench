"""Run a command with nnWorkbench's saved nnU-Net environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath

from .json_output import print_json, stdout_to_stderr, use_utf8_output
from .path_setup import configured_paths, note_environment_differences
from .services.checkpoint_reduction import reduce_model_for_sharing, reduction_plan
from .services.commands import shell_join
from .services.dataset_scanner import discover
from .services.image_stack import stack_image_folder
from .services.label_analysis import analyse_label_distribution, analysis_path
from .services.process_runner import crash_note, exit_status

BUILTIN_COMMANDS = (
    "list",
    "status",
    "suggest",
    "validation",
    "create-dataset",
    "describe",
    "new-iteration",
    "compute-statistics",
    "inspect",
    "annotate",
    "inspect-predictions",
    "stack-images",
    "reduce-checkpoints",
)


def _progress(as_json: bool):
    """Progress messages: stdout normally, stderr with --json (so stdout stays valid JSON)."""
    return (lambda message: print(message, file=sys.stderr)) if as_json else print


def find_executable(name: str) -> str:
    """Find a console script such as nnUNetv2_train: next to this Python first (bin/ on Linux and macOS, Scripts\\ on
    Windows), then on PATH. This environment's nnU-Net wins over another one earlier on PATH (e.g. conda base's), also
    when the environment is not activated (a Windows shortcut, the macOS Dock, an agent calling the program by its
    full path): another nnU-Net version may not know nnWorkbench's trainers (nnUNet_extTrainer)."""
    folder = Path(sys.executable).parent
    for directory in (folder, folder / "Scripts", folder / "bin"):
        found = shutil.which(name, path=str(directory))
        if found:
            return found
    return shutil.which(name) or name


def is_program(command: str, name: str, windows: bool = os.name == "nt") -> bool:
    """Whether *command* (a name or path, as typed) starts the program *name*: on Windows also in another case and
    with .exe, as Windows finds it then too."""
    if not windows:
        return PurePosixPath(command).name == name
    return PureWindowsPath(command).name.casefold().removesuffix(".exe") == name.casefold()


YES_ENVIRONMENT = "NNWORKBENCH_YES"  # set by the GUI, which asks the user itself before such jobs


def trained_fold(command: Sequence[str], results_root: Path | None) -> tuple[Path, argparse.Namespace] | None:
    """The fold folder (with a log or checkpoint) an nnUNetv2_train command trains, and its arguments."""
    # any spelling that starts nnU-Net's training: nnunetv2_train and nnUNetv2_train.exe also run it on Windows
    if not command or not is_program(command[0], "nnUNetv2_train") or results_root is None or len(command) < 4:
        return None
    parsed = _training_arguments(command[1:])
    if parsed is None:
        return None
    dataset, configuration, trainer, plans = parsed.dataset, parsed.configuration, parsed.tr, parsed.p
    fold = parsed.fold if parsed.fold == "all" or not parsed.fold.isdigit() else str(int(parsed.fold))
    folders = ([path for path in results_root.glob(f"Dataset{int(dataset):03d}_*")] if dataset.isdigit()
               else [results_root / dataset])
    for folder in folders:
        fold_folder = folder / f"{trainer}__{plans}__{configuration}" / f"fold_{fold}"
        if fold_folder.is_dir() and (any(fold_folder.glob("training_log_*.txt"))
                                     or any(fold_folder.glob("checkpoint_*.pth"))):
            return fold_folder, parsed
    return None


def existing_training(command: Sequence[str], results_root: Path | None) -> Path | None:
    """The fold folder an nnUNetv2_train command would start again from scratch, or None.

    nnUNetv2_train DATASET CONFIGURATION FOLD [-tr TRAINER] [-p PLANS] replaces an existing fold unless --c
    (continue) or --val (validation only) is given.
    """
    found = trained_fold(command, results_root)
    return found[0] if found and not (found[1].c or found[1].val) else None


def running_training(command: Sequence[str], results_root: Path | None) -> Path | None:
    """The fold folder of an nnUNetv2_train command whose fold seems to be still training (log written lately)."""
    from .services.workflow_status import fold_info

    found = trained_fold(command, results_root)
    return found[0] if found and fold_info(found[0]).status == "training" else None


class _ArgumentError(Exception):
    pass


class _TrainingParser(argparse.ArgumentParser):
    def error(self, message: str):
        raise _ArgumentError(message)


def _training_arguments(arguments: Sequence[str]) -> argparse.Namespace | None:
    """nnUNetv2_train's arguments read as nnU-Net reads them (run_training.py): options may come before the
    positional ones and as -tr=X; unknown options are ignored. None when they cannot be read."""
    parser = _TrainingParser(add_help=False, allow_abbrev=False)
    parser.add_argument("dataset")
    parser.add_argument("configuration")
    parser.add_argument("fold")
    parser.add_argument("-tr", default="nnUNetTrainer")
    parser.add_argument("-p", default="nnUNetPlans")
    for name in ("-pretrained_weights", "-num_gpus", "-device"):
        parser.add_argument(name)
    parser.add_argument("--c", action="store_true")
    parser.add_argument("--val", action="store_true")
    try:
        return parser.parse_known_args(list(arguments))[0]
    except _ArgumentError:
        return None


def run_with_configured_paths(command: Sequence[str]) -> int:
    """Run *command* with saved nnU-Net paths, returning its exit status.

    --yes confirms replacing an existing training; --dry-run shows what would run and changes nothing.
    """
    if not command:
        raise ValueError("Specify a command to run.")
    confirmed = "--yes" in command or os.environ.get(YES_ENVIRONMENT) == "1"
    dry_run = "--dry-run" in command
    command = [part for part in command if part not in ("--yes", "--dry-run")]
    paths = configured_paths()
    note_environment_differences(paths)
    results_root = Path(paths["nnUNet_results"]) if paths.get("nnUNet_results") else None
    replaced = existing_training(command, results_root)
    running = running_training(command, results_root)
    if dry_run:
        print("Would run: " + shell_join([find_executable(command[0]), *command[1:]]))
        for name, value in paths.items():
            print(f"  {name}={value}")
        if replaced is not None:
            print(f"  This starts {replaced} again from scratch and replaces its results"
                  + ("" if confirmed else " (needs --yes)") + ".")
        if running is not None:
            print(f"  {running} seems to be still training" + ("" if confirmed else " (needs --yes)") + ".")
        return 0
    if replaced is not None and not confirmed:
        raise ValueError(f"{replaced} already has a training. Running this command starts it again and replaces its "
                         "checkpoints and validation results. Add --c to continue it instead, or --yes to start it "
                         "again.")
    if running is not None and not confirmed:
        raise ValueError(f"{running} seems to be still training (its log changed lately). A second run of the same "
                         "fold writes into the same folder. Check that the other run has ended (its job or process); "
                         "then add --yes.")
    program = find_executable(command[0])
    if program == command[0] and shutil.which(program) is None:  # found neither on PATH nor next to Python
        raise FileNotFoundError(
            f"'{command[0]}' is neither an nnWorkbench tool ({', '.join(BUILTIN_COMMANDS)}) nor an installed "
            "program such as nnUNetv2_train. Add --help to a tool to see its options.")
    environment = os.environ.copy()
    environment.update(paths)
    if os.name == "nt":
        # nnU-Net reads dataset.json and plans with open()'s default encoding, the ANSI code page (cp1252) on
        # Windows: UTF-8 mode reads them as UTF-8, as they are written (also by other tools than nnWorkbench)
        environment.setdefault("PYTHONUTF8", "1")
    code = subprocess.run([program, *command[1:]], env=environment, check=False).returncode
    note = crash_note(program, code)
    if note:
        print(note, file=sys.stderr)
    return exit_status(code)  # signal N: 128 + N as shells do; a Windows crash status: passed on unchanged


def saved_roots() -> dict[str, Path | None]:
    """The saved nnU-Net folders (None when a folder is not set)."""
    paths = configured_paths()
    note_environment_differences(paths)
    return {key: Path(paths[f"nnUNet_{key}"]) if paths.get(f"nnUNet_{key}") else None
            for key in ("raw", "preprocessed", "results")}


def raw_dataset_directory(dataset_id: int) -> Path:
    """Resolve a dataset ID below the configured Raw root."""
    raw_root = Path(configured_paths()["nnUNet_raw"])
    datasets = discover(raw_root)
    if dataset_id in datasets:
        return datasets[dataset_id][1]
    raise ValueError(f"No Raw dataset matching ID {dataset_id} was found in {raw_root}.")


def run_compute_statistics(dataset_id: int, workers: int | None = None, as_json: bool = False) -> int:
    """Compute statistics (as in the GUI): classes per image and voxels per class, from labelsTr."""
    dataset_directory = raw_dataset_directory(dataset_id)
    analysis = analyse_label_distribution(dataset_directory, workers, progress=_progress(as_json))
    result_path = analysis_path(dataset_directory)
    if as_json:
        print_json({"dataset": dataset_directory.name, "path": str(result_path)} | analysis)
    else:
        print(f"Statistics computed: {result_path} ({analysis['files_analyzed']} file(s))")
    return 0


def run_image_stack(input_directory: Path, *, file_type: str, output_path: Path | None = None,
                    workers: int | None = None, scale_factor: int = 1,
                    spacing: tuple[float, float, float] | None = None, overwrite: bool = False,
                    show_mitk_output: bool = False, as_json: bool = False) -> int:
    """Stack 2D image slices into one NIfTI volume and print concise progress."""
    output = stack_image_folder(
        input_directory,
        file_type=file_type,
        output_path=output_path,
        workers=workers,
        scale_factor=scale_factor,
        spacing=spacing,
        overwrite=overwrite,
        quiet=not show_mitk_output,
        progress=_progress(as_json),
    )
    if as_json:
        print_json({"input": str(input_directory), "output": str(output), "file_type": file_type,
                    "scale_factor": scale_factor, "spacing": list(spacing) if spacing else [1.0, 1.0, 1.0]})
    return 0


def run_checkpoint_reduction(model_directory: Path, *, reset_trainer_name: bool = False,
                             as_json: bool = False, yes: bool = False, dry_run: bool = False) -> int:
    """Strip a trained model folder down to inference-only files and print progress.

    Changes the model in place and cannot be undone: needs ``yes``; ``dry_run`` only lists the changes.
    """
    plan = reduction_plan(model_directory, reset_trainer_name=reset_trainer_name)
    if dry_run or not yes:
        refusal = ("Add --yes to reduce the model (work on a copy if you still need the validation results or want "
                   "to continue training).")
        if as_json:  # one JSON document: the plan, and without --yes the refusal in "error" (exit code 1)
            print_json({"model": str(model_directory), "dry_run": True,
                        "actions": [{"action": action, "path": str(path)} for action, path in plan],
                        **({} if dry_run else {"error": refusal})})
            return 0 if dry_run else 1
        print(f"{'Would change' if dry_run else 'This changes'} {model_directory} in place (cannot be undone):")
        for action, path in plan:
            print(f"  {action}: {path}")
        if dry_run:
            return 0
        raise ValueError(refusal)
    folds = reduce_model_for_sharing(
        model_directory, reset_trainer_name=reset_trainer_name, progress=_progress(as_json),
    )
    if as_json:
        print_json({"model": str(model_directory), "folds": [fold.name for fold in folds],
                    "trainer_name_reset": reset_trainer_name})
    else:
        print(f"Reduced {len(folds)} fold(s) in {model_directory} for sharing.")
    return 0


def _case_image(dataset_id: int, index: int = 1, case: str = "") -> tuple[Path, Path, int, int]:
    """The first channel image of a case, chosen by name or by its position in filename order."""
    from .services.napari_inspector import image_case_ids, related_files_for_case

    dataset_directory = raw_dataset_directory(dataset_id)
    cases = image_case_ids(dataset_directory)
    if not cases:
        raise ValueError(f"No images were found in {dataset_directory / 'imagesTr'} (or the files dataset.json "
                         "lists).")
    if case:
        if case not in cases:
            raise ValueError(f"No case '{case}' in {dataset_directory.name} (cases: {cases[0]} ... {cases[-1]}).")
        index = cases.index(case) + 1
    elif not 1 <= index <= len(cases):
        raise ValueError(f"Image index must be between 1 and {len(cases)}.")
    image = related_files_for_case(dataset_directory, cases[index - 1]).images[0]
    return dataset_directory, image, index, len(cases)


def run_dataset_inspection(dataset_id: int, index: int = 1, case: str = "") -> int:
    """Open a Raw dataset's images and labels in Napari, starting at a case (by name or index)."""
    try:
        # Imported lazily: the lightweight remote installation has no Qt or Napari.
        from .services.napari_inspector import open_case
    except ImportError as error:
        raise RuntimeError("Napari is not installed. Use the full installation to inspect datasets.") from error
    dataset_directory, image, index, total = _case_image(dataset_id, index, case)
    print(f"Inspecting {image.name} ({index} of {total}) in {dataset_directory.name}")
    open_case(image)
    return 0


def run_annotation(dataset_id: int, index: int = 1, case: str = "") -> int:
    """Open a case for annotating in Napari, as the pencil in the GUI (saving writes labelsTr)."""
    try:
        from .services.napari_inspector import open_annotation
    except ImportError as error:
        raise RuntimeError("Napari is not installed. Use the full installation to annotate.") from error
    from .services.dataset_json import REGIONS_NOTE, has_regions, read_manifest

    dataset_directory, image, index, total = _case_image(dataset_id, index, case)
    try:
        manifest = read_manifest(dataset_directory)
    except (OSError, ValueError):
        manifest = None
    if has_regions(manifest):
        raise ValueError(f"{dataset_directory.name} has region-based labels. {REGIONS_NOTE}")
    print(f"Annotating {image.name} ({index} of {total}) in {dataset_directory.name}. "
          "Click Save annotation in Napari, then close it.")
    open_annotation(image)
    return 0


def _run_builtin_command(command: Sequence[str]) -> int | None:
    """Run an nnWorkbench data-management command, or return ``None`` for nnU-Net."""
    if not command:
        return None
    name, arguments = command[0], command[1:]
    if name == "compute-statistics":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Compute statistics (as Compute statistics in the GUI): in how many images each class "
                        "occurs and how many voxels it covers, read from labelsTr.")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("--workers", type=int, default=None, help="Worker processes (as in the GUI settings)")
        parser.add_argument("--json", action="store_true",
                            help="Print the statistics as JSON (progress goes to stderr), e.g. for scripts")
        options = parser.parse_args(arguments)
        return run_compute_statistics(options.dataset, options.workers, options.json)
    if name == "inspect":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Open a dataset's images and labels in Napari.")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("-i", "--index", type=int, default=1,
                            help="Image to start with, counting from 1 in filename order (default: 1)")
        parser.add_argument("--case", default="", help="Case to start with, e.g. cells_005 (instead of --index)")
        options = parser.parse_args(arguments)
        return run_dataset_inspection(options.dataset, options.index, options.case)
    if name == "inspect-predictions":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Open predictions in Napari, each with its input image (as on the Predict page).")
        parser.add_argument("predictions", type=Path, help="Output folder of nnUNetv2_predict")
        parser.add_argument("--images", type=Path, default=None, help="Input folder with the predicted images")
        options = parser.parse_args(arguments)
        if not options.predictions.is_dir():
            raise ValueError(f"Prediction folder not found: {options.predictions}")
        try:
            from .services.napari_inspector import open_prediction_folder
        except ImportError as error:
            raise RuntimeError("Napari is not installed. Use the full installation to inspect predictions.") from error
        print(f"Opening the predictions in {options.predictions}")
        open_prediction_folder(options.predictions, options.images)
        return 0
    if name == "annotate":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Open a case for annotating in Napari with nnInteractive (as the pencil in the GUI). "
                        "Save annotation writes the label to labelsTr; sparse labels are shifted while annotating "
                        "and saved with the original IDs.")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("--case", default="", help="Case to annotate, e.g. cells_005")
        parser.add_argument("-i", "--index", type=int, default=1,
                            help="Or the case by position, counting from 1 in filename order (default: 1)")
        options = parser.parse_args(arguments)
        return run_annotation(options.dataset, options.index, options.case)
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
        parser.add_argument("--json", action="store_true",
                            help="Print the result as JSON (progress goes to stderr), e.g. for scripts")
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
            as_json=options.json,
        )
    if name == "reduce-checkpoints":
        parser = argparse.ArgumentParser(prog=f"nnworkbench-run {name}")
        parser.add_argument("model_directory", type=Path,
                            help="Trained model folder containing fold_* subdirectories")
        parser.add_argument("--reset-trainer-name", action="store_true",
                            help="Rewrite each checkpoint's trainer_name to nnUNetTrainer")
        parser.add_argument("--yes", action="store_true",
                            help="Confirm: the model is changed in place and this cannot be undone")
        parser.add_argument("--dry-run", action="store_true", help="Only list what would be changed or deleted")
        parser.add_argument("--json", action="store_true",
                            help="Print the result as JSON (progress goes to stderr), e.g. for scripts")
        options = parser.parse_args(arguments)
        return run_checkpoint_reduction(
            options.model_directory, reset_trainer_name=options.reset_trainer_name, as_json=options.json,
            yes=options.yes, dry_run=options.dry_run,
        )
    if name == "list":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="List the datasets in the saved nnU-Net folders: ID, name, where they exist, ID problems "
                        "and the iteration (from the _iterN name).")
        parser.add_argument("--json", action="store_true", help="Print the list as JSON, e.g. for scripts")
        options = parser.parse_args(arguments)
        from .cli_tools import run_list

        return run_list(saved_roots(), as_json=options.json)
    if name == "status":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Show what is done and what is next for a dataset: checks, preprocessing, models and folds "
                        "(the same states as the stage tabs in the GUI).")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("--json", action="store_true", help="Print the status as JSON, e.g. for scripts")
        parser.add_argument("--no-device", action="store_true",
                            help="Skip the device check (GPU / MPS / CPU), which takes a few seconds")
        options = parser.parse_args(arguments)
        from .status_report import run_status

        return run_status(options.dataset, saved_roots(), as_json=options.json, check_device=not options.no_device)
    if name == "validation":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Validation results (as the Validate page in the GUI): Dice per fold and class, and the "
                        "validation cases sorted from weakest to best.")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("--model", default="", help="Only this model, or the models whose name contains this "
                                                        "text, e.g. nnUNetTrainerBetterIgnoreSampling")
        parser.add_argument("--fold", default="", help="Only this fold, e.g. 0 (or fold_0) or all")
        parser.add_argument("--cases", type=int, default=10,
                            help="Weakest cases to list per model in the text output (0 = all; JSON lists all)")
        parser.add_argument("--json", action="store_true", help="Print the results as JSON, e.g. for scripts")
        parser.add_argument("--open", default="", metavar="CASE",
                            help="Open this validation case in Napari: image, label and prediction")
        options = parser.parse_args(arguments)
        if options.open and options.json:  # as the other tools that only open Napari
            parser.error("--open opens Napari and prints no result: leave out --json")
        from .cli_tools import run_validation

        return run_validation(saved_roots(), options.dataset, model=options.model, fold=options.fold,
                              limit=options.cases, as_json=options.json, open_case=options.open)
    if name == "suggest":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Print the command the GUI would prefill for a step (same defaults), and why.")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("--step", required=True, choices=("preprocess", "train", "predict"), help="The step")
        parser.add_argument("--model", default="", help="predict: only models whose name contains this text")
        parser.add_argument("--input", default="", help="predict: folder with the images to predict")
        parser.add_argument("--output", default="", help="predict: folder for the predictions")
        parser.add_argument("--no-device", action="store_true",
                            help="Skip the device check (assume an NVIDIA GPU, nnU-Net's default)")
        parser.add_argument("--json", action="store_true", help="Print the suggestion as JSON, e.g. for scripts")
        options = parser.parse_args(arguments)
        from .cli_tools import run_suggest

        return run_suggest(saved_roots(), options.dataset, options.step, check_device=not options.no_device,
                           model=options.model, input_directory=options.input, output_directory=options.output,
                           as_json=options.json)
    if name == "create-dataset":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Create a dataset in nnUNet_raw (as New dataset in the GUI): imagesTr, labelsTr and a "
                        "complete dataset.json.")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="New dataset ID, e.g. 42")
        parser.add_argument("--name", required=True, help="Dataset name, e.g. CellSegmentation")
        parser.add_argument("--labels", required=True,
                            help="Classes in ID order, comma-separated, e.g. background,cell,nucleus "
                                 "(background is added as 0 if missing)")
        parser.add_argument("--ignore", action="store_true",
                            help="Add the ignore label with the highest ID (sparse labels)")
        parser.add_argument("--channels", default="image",
                            help="Channel names in order, comma-separated, e.g. fluorescence (default: image)")
        parser.add_argument("--file-ending", default=".nii.gz", help="File ending of all images (default: .nii.gz)")
        parser.add_argument("--description", default="", help="Description for dataset.json")
        parser.add_argument("--add-description", action="store_true",
                            help="Also create dataset-description.md, a template to describe the images and classes")
        parser.add_argument("--json", action="store_true", help="Print the result as JSON, e.g. for scripts")
        options = parser.parse_args(arguments)
        from .cli_tools import run_create_dataset

        return run_create_dataset(saved_roots(), options.dataset, options.name, options.labels.split(","),
                                  options.channels.split(","), ignore=options.ignore, file_ending=options.file_ending,
                                  description=options.description, add_description=options.add_description,
                                  as_json=options.json)
    if name == "describe":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Create dataset-description.md in the dataset folder: a template to describe the images and "
                        "define the classes (guide: 1. Describe, 2. Define). An existing file is never replaced.")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Dataset ID")
        parser.add_argument("--json", action="store_true", help="Print the result as JSON, e.g. for scripts")
        options = parser.parse_args(arguments)
        from .cli_tools import run_describe

        return run_describe(saved_roots(), options.dataset, as_json=options.json)
    if name == "new-iteration":
        parser = argparse.ArgumentParser(
            prog=f"nnworkbench-run {name}",
            description="Start the next iteration (as + New iteration in the GUI): copy imagesTr, labelsTr "
                        "and dataset.json of a dataset into a new dataset.")
        parser.add_argument("-d", "--dataset", type=int, required=True, help="Source dataset ID")
        parser.add_argument("--id", type=int, default=None,
                            help="ID of the new dataset (default: the next free ID after the source)")
        parser.add_argument("--name", default=None, help="Name of the new dataset (default: <name>_iter1, _iter2, ...)")
        parser.add_argument("--link-images", action="store_true",
                            help="Do not copy the images: dataset.json of the new dataset lists the source's images "
                                 "(\"dataset\" file list). Labels are always copied.")
        parser.add_argument("--json", action="store_true", help="Print the result as JSON, e.g. for scripts")
        options = parser.parse_args(arguments)
        from .cli_tools import run_new_iteration

        return run_new_iteration(saved_roots(), options.dataset, options.id, options.name,
                                 link_images=options.link_images, as_json=options.json)
    return None


def main() -> None:
    """Run a command after applying the persisted nnWorkbench path configuration."""
    use_utf8_output()  # also for --help: piped output on Windows is cp1252 otherwise
    parser = argparse.ArgumentParser(
        description="Run nnU-Net commands or nnWorkbench dataset tools with saved paths.",
        epilog=f"nnWorkbench tools: {', '.join(BUILTIN_COMMANDS)}. Each takes --help. Any other command (e.g. "
               "nnUNetv2_train) runs with the saved nnU-Net folders; --dry-run shows it without running.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="nnU-Net command or nnWorkbench subcommand")
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    as_json = bool(command[:1]) and command[0] in BUILTIN_COMMANDS and "--json" in command
    if as_json:  # only the JSON result on stdout; progress and library logs go to stderr
        with stdout_to_stderr():
            try:
                raise SystemExit(_run_builtin_command(command))
            except (RuntimeError, TypeError, ValueError, OSError) as error:
                print_json({"error": str(error)})  # errors stay machine-readable too
                raise SystemExit(1) from error
            except Exception as error:  # e.g. a corrupt checkpoint: still JSON, not a traceback
                print_json({"error": f"{type(error).__name__}: {error}"})
                raise SystemExit(1) from error
    try:
        builtin_exit_code = _run_builtin_command(command)
        raise SystemExit(builtin_exit_code if builtin_exit_code is not None else run_with_configured_paths(command))
    except (RuntimeError, TypeError, ValueError, OSError) as error:  # OSError: e.g. a read-only nnUNet_raw
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
