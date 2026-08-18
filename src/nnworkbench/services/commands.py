from __future__ import annotations

import shlex
from pathlib import Path


def preprocessing_command(dataset_id: int, configuration: str, planner: str) -> str:
    return shlex.join(("nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-c", configuration,
                       "-pl", planner))


def training_command(dataset_id: int, plans_file: str, configuration: str, trainer: str, fold: str) -> str:
    return shlex.join(("nnUNetv2_train", str(dataset_id), configuration, fold, "-tr", trainer,
                       "-p", Path(plans_file).stem))


def prediction_command(dataset_id: int, input_directory: str, output_directory: str, plans: str,
                       configuration: str) -> str:
    return shlex.join(("nnUNetv2_predict", "-i", input_directory, "-o", output_directory, "-d",
                       str(dataset_id), "-c", configuration, "-p", plans))
