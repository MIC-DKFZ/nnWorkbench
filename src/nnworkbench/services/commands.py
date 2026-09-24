from __future__ import annotations

import shlex
from pathlib import Path


def preprocessing_command(dataset_id: int, configuration: str, planner: str, additional_arguments: str = "") -> str:
    return shlex.join(("nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-c", configuration,
                       "-pl", planner, "--verify_dataset_integrity", *shlex.split(additional_arguments)))


def training_command(dataset_id: int, plans_file: str, configuration: str, trainer: str, fold: str,
                     additional_arguments: str = "") -> str:
    return shlex.join(("nnUNetv2_train", str(dataset_id), configuration, fold, "-tr", trainer,
                       "-p", Path(plans_file).stem, *shlex.split(additional_arguments)))


def prediction_command(dataset_id: int, input_directory: str, output_directory: str, plans: str,
                       configuration: str, additional_arguments: str = "", trainer: str = "nnUNetTrainer",
                       folds: str = "0 1 2 3 4") -> str:
    arguments = ["nnUNetv2_predict", "-i", input_directory, "-o", output_directory, "-d",
                 str(dataset_id), "-c", configuration, "-p", plans, "-tr", trainer]
    if selected_folds := shlex.split(folds):
        arguments.extend(("-f", *selected_folds))
    return shlex.join((*arguments, *shlex.split(additional_arguments)))


def wrapped_command(command: str) -> str:
    """Prefix an nnU-Net command with the saved-path nnWorkbench runner."""
    return shlex.join(("nnworkbench-run", *shlex.split(command)))
