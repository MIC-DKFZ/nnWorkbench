"""Strip a trained nnU-Net model folder down to what inference needs."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import torch

from .dataset_contents import model_folds

DEFAULT_TRAINER_NAME = "nnUNetTrainer"


def reduce_checkpoint(checkpoint_path: Path | str, *, reset_trainer_name: bool = False) -> None:
    """Drop the optimizer state from one checkpoint file, saving it back in place.

    When ``reset_trainer_name`` is set, the checkpoint's ``trainer_name`` is also
    rewritten to ``nnUNetTrainer`` so custom trainer classes are not required for inference.
    """
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer_state", None)
    if reset_trainer_name:
        checkpoint["trainer_name"] = DEFAULT_TRAINER_NAME
    torch.save(checkpoint, path)


def reduce_model_for_sharing(
    model_path: Path | str, *, reset_trainer_name: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Strip every fold of a trained nnU-Net model folder down to inference-only files.

    For each fold, ``checkpoint_final.pth`` has its optimizer state removed (and its
    ``trainer_name`` reset when ``reset_trainer_name`` is set), while ``checkpoint_best.pth``
    and the ``validation`` directory are deleted entirely. Returns the processed fold paths.
    """
    directory = Path(model_path)
    folds = model_folds(directory)
    if not folds:
        raise ValueError(f"No fold_* directories were found in {directory}.")
    for fold in folds:
        final_checkpoint = fold / "checkpoint_final.pth"
        if final_checkpoint.is_file():
            reduce_checkpoint(final_checkpoint, reset_trainer_name=reset_trainer_name)
            if progress is not None:
                progress(f"Reduced {final_checkpoint}")
        best_checkpoint = fold / "checkpoint_best.pth"
        if best_checkpoint.is_file():
            best_checkpoint.unlink()
            if progress is not None:
                progress(f"Deleted {best_checkpoint}")
        validation_directory = fold / "validation"
        if validation_directory.is_dir():
            shutil.rmtree(validation_directory)
            if progress is not None:
                progress(f"Deleted {validation_directory}")
    return folds
