"""Strip a trained nnU-Net model folder down to what inference needs."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from .dataset_contents import model_folds

DEFAULT_TRAINER_NAME = "nnUNetTrainer"


def reduce_checkpoint(checkpoint_path: Path | str, *, reset_trainer_name: bool = False) -> None:
    """Drop the optimizer state from one checkpoint file, saving it back in place.

    When ``reset_trainer_name`` is set, the checkpoint's ``trainer_name`` is also
    rewritten to ``nnUNetTrainer`` so custom trainer classes are not required for inference.
    """
    # imported here: every nnworkbench-run call imports this module, and torch takes ~2 s to import (and a broken
    # torch installation would otherwise break `status`, the command that reports it)
    import torch

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer_state", None)
    if reset_trainer_name:
        checkpoint["trainer_name"] = DEFAULT_TRAINER_NAME
    torch.save(checkpoint, path)


def reduction_plan(model_path: Path | str, *, reset_trainer_name: bool = False) -> list[tuple[str, Path]]:
    """What reduce_model_for_sharing would do, as (action, path) pairs; changes nothing."""
    directory = Path(model_path)
    folds = model_folds(directory)
    if not folds:
        raise ValueError(f"No fold_* directories were found in {directory}.")
    actions: list[tuple[str, Path]] = []
    for fold in folds:
        if (fold / "checkpoint_final.pth").is_file():
            actions.append(("reduce" + (" and reset trainer name" if reset_trainer_name else ""),
                            fold / "checkpoint_final.pth"))
        if (fold / "checkpoint_best.pth").is_file():
            actions.append(("delete", fold / "checkpoint_best.pth"))
        if (fold / "validation").is_dir():
            actions.append(("delete folder", fold / "validation"))
    return actions


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
    plan = reduction_plan(directory, reset_trainer_name=reset_trainer_name)
    for action, path in plan:
        if action.startswith("reduce"):
            reduce_checkpoint(path, reset_trainer_name=reset_trainer_name)
            if progress is not None:
                progress(f"Reduced {path}")
        elif action == "delete":
            path.unlink()
            if progress is not None:
                progress(f"Deleted {path}")
        else:
            shutil.rmtree(path)
            if progress is not None:
                progress(f"Deleted {path}")
    return model_folds(directory)
