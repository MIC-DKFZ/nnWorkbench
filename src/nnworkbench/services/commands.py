from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path


def preprocessing_command(dataset_id: int, configuration: str | Sequence[str], planner: str,
                          additional_arguments: str = "", *, processes: int | None = None,
                          verify: bool = True) -> str:
    configurations = [configuration] if isinstance(configuration, str) else list(configuration)
    arguments = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-c", *configurations, "-pl", planner]
    if processes:
        arguments += ["-np", str(processes)]
    if verify:
        arguments.append("--verify_dataset_integrity")
    return shlex.join((*arguments, *shlex.split(additional_arguments)))


def training_command(dataset_id: int, plans_file: str, configuration: str, trainer: str, fold: str,
                     additional_arguments: str = "", *, continue_training: bool = False) -> str:
    arguments = ["nnUNetv2_train", str(dataset_id), configuration, fold, "-tr", trainer, "-p", Path(plans_file).stem]
    if continue_training:
        arguments.append("--c")
    return shlex.join((*arguments, *shlex.split(additional_arguments)))


def prediction_command(dataset_id: int, input_directory: str, output_directory: str, plans: str,
                       configuration: str, additional_arguments: str = "", trainer: str = "nnUNetTrainer",
                       folds: str = "0 1 2 3 4", *, disable_tta: bool = False,
                       continue_prediction: bool = False) -> str:
    arguments = ["nnUNetv2_predict", "-i", input_directory, "-o", output_directory, "-d",
                 str(dataset_id), "-c", configuration, "-p", plans, "-tr", trainer]
    if selected_folds := shlex.split(folds):
        arguments.extend(("-f", *selected_folds))
    if disable_tta:
        arguments.append("--disable_tta")
    if continue_prediction:
        arguments.append("--continue_prediction")  # skip images that already have a prediction
    return shlex.join((*arguments, *shlex.split(additional_arguments)))


def wrapped_command(command: str) -> str:
    """Prefix an nnU-Net command with the saved-path nnWorkbench runner."""
    return shlex.join(("nnworkbench-run", *shlex.split(command)))
