from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

# Commands are kept as POSIX-quoted strings inside nnWorkbench (created with shlex.join, so Windows paths survive).
# What the user sees or copies uses the quoting of the platform's shell.
WINDOWS = os.name == "nt"


def split_arguments(text: str) -> list[str]:
    """Split additional arguments typed by the user; on Windows backslashes in paths are kept."""
    if WINDOWS:
        return [part[1:-1] if len(part) > 1 and part[0] == part[-1] == '"' else part
                for part in shlex.split(text, posix=False)]
    return shlex.split(text)


def shell_join(arguments: Sequence[str]) -> str:
    """Quote for the platform's shell: cmd/PowerShell on Windows, a POSIX shell elsewhere."""
    return subprocess.list2cmdline(list(arguments)) if WINDOWS else shlex.join(arguments)


def preprocessing_command(dataset_id: int, configuration: str | Sequence[str], planner: str,
                          additional_arguments: str = "", *, processes: int | None = None,
                          verify: bool = True) -> str:
    configurations = [configuration] if isinstance(configuration, str) else list(configuration)
    arguments = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-c", *configurations, "-pl", planner]
    if processes:
        arguments += ["-np", str(processes)]
    if verify:
        arguments.append("--verify_dataset_integrity")
    return shlex.join((*arguments, *split_arguments(additional_arguments)))


def training_command(dataset_id: int, plans_file: str, configuration: str, trainer: str, fold: str,
                     additional_arguments: str = "", *, continue_training: bool = False,
                     device: str | None = None) -> str:
    arguments = ["nnUNetv2_train", str(dataset_id), configuration, fold, "-tr", trainer, "-p", Path(plans_file).stem]
    if device:  # None keeps nnU-Net's default (cuda)
        arguments += ["-device", device]
    if continue_training:
        arguments.append("--c")
    return shlex.join((*arguments, *split_arguments(additional_arguments)))


def prediction_command(dataset_id: int, input_directory: str, output_directory: str, plans: str,
                       configuration: str, additional_arguments: str = "", trainer: str = "nnUNetTrainer",
                       folds: str | Sequence[str] = "0 1 2 3 4", *, disable_tta: bool = False,
                       continue_prediction: bool = False, preprocessing_processes: int = 3,
                       export_processes: int = 3, device: str | None = None) -> str:
    arguments = ["nnUNetv2_predict", "-i", input_directory, "-o", output_directory, "-d",
                 str(dataset_id), "-c", configuration, "-p", plans, "-tr", trainer]
    # a list of fold names as they are; a string is split at spaces ("0 1 2")
    if selected_folds := folds.split() if isinstance(folds, str) else list(folds):
        arguments.extend(("-f", *selected_folds))
    # processes for preprocessing and for exporting the segmentations (nnU-Net's default: 3 each)
    arguments += ["-npp", str(preprocessing_processes), "-nps", str(export_processes)]
    if device:  # None keeps nnU-Net's default (cuda)
        arguments += ["-device", device]
    if disable_tta:
        arguments.append("--disable_tta")
    if continue_prediction:
        arguments.append("--continue_prediction")  # skip images that already have a prediction
    return shlex.join((*arguments, *split_arguments(additional_arguments)))


def wrapped_command(command: str) -> str:
    """Prefix an nnU-Net command with the saved-path nnWorkbench runner, quoted for the platform's shell."""
    return shell_join(("nnworkbench-run", *shlex.split(command)))
