"""The defaults the GUI prefills, in one place for the GUI and `nnworkbench-run suggest` (Qt-free)."""

from __future__ import annotations

import os
from collections.abc import Iterable

DEFAULT_PLANNER = "nnUNetPlannerResEncL"
PLANNERS = ("ExperimentPlanner", "nnUNetPlannerResEncM", DEFAULT_PLANNER, "nnUNetPlannerResEncXL")
DEFAULT_PLANS = "nnUNetResEncUNetLPlans"  # written by the default planner
CONFIGURATIONS = ("3d_fullres", "2d", "3d_lowres")
DEFAULT_CONFIGURATION = "3d_fullres"
SPARSE_TRAINER = "nnUNetTrainerBetterIgnoreSampling"
DENSE_TRAINER = "nnUNetTrainer"
PREDICTION_PROCESSES = 3  # nnU-Net's default for -npp and -nps


def preprocessing_processes() -> int:
    """Processes for preprocessing (-np): up to 8, never more than CPU cores."""
    return min(8, os.cpu_count() or 1)


def prediction_processes() -> int:
    """Processes for prediction (-npp and -nps): nnU-Net's default of 3, never more than CPU cores (the GUI's
    fields cannot go beyond the cores, so `suggest` gives the same number on a small machine)."""
    return min(PREDICTION_PROCESSES, max(1, os.cpu_count() or 1))


def trainer_for(sparse: bool) -> str:
    """Sparse labels (dataset.json has an ignore label) train with the bundled trainer, dense labels with nnU-Net's."""
    return SPARSE_TRAINER if sparse else DENSE_TRAINER


def default_plans(available: Iterable[str]) -> str:
    """The default plans when preprocessed, otherwise the first preprocessed plans ("" if none)."""
    names = list(available)
    return DEFAULT_PLANS if DEFAULT_PLANS in names else (names[0] if names else "")


def preprocessing_configuration(manifest: object, planned: Iterable[str] = ()) -> str:
    """2d for 2D images, otherwise the default. nnU-Net plans no 3d_fullres for 2D images and then skips it
    ("Configuration 3d_fullres not found") without an error. 2D: .png/.bmp, or plans (``planned``: the configuration
    names of the existing plans files) without 3d_fullres but with 2d (e.g. 2D .tif)."""
    ending = manifest.get("file_ending") if isinstance(manifest, dict) else None
    planned = set(planned)
    if isinstance(ending, str) and ending.lower() in (".png", ".bmp") or "2d" in planned and DEFAULT_CONFIGURATION not in planned:
        return "2d"
    return DEFAULT_CONFIGURATION


def default_configuration(available: Iterable[str]) -> str:
    """3d_fullres when available, otherwise the first available configuration ("" if none)."""
    names = list(available)
    return DEFAULT_CONFIGURATION if DEFAULT_CONFIGURATION in names else (names[0] if names else "")


def next_fold(fold_count: int, existing: Iterable[str]) -> str | None:
    """The first fold of splits_final.json, then "all", that the model does not have yet (None if all exist)."""
    taken = set(existing)
    return next((fold for fold in [str(index) for index in range(fold_count)] + ["all"] if fold not in taken), None)


def prediction_folds(finished: Iterable[str]) -> list[str]:
    """The finished cross-validation folds; fold "all" only when it is the only finished fold."""
    names = list(finished)
    return [name for name in names if name != "all"] or names
