from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

WORKBENCH_FOLDER = "nnworkbench"  # in a raw dataset: what nnWorkbench adds (nnU-Net ignores it)
# nnU-Net names its fold folders fold_0, fold_1, … and fold_all; other folders are not folds (-f would reject them)
_FOLD_FOLDER = re.compile(r"fold_(\d+|all)")


def is_fold_folder(path: Path) -> bool:
    return _FOLD_FOLDER.fullmatch(path.name) is not None and path.is_dir()


def workbench_folder(dataset_directory: Path) -> Path:
    """The folder of a raw dataset with nnWorkbench's files: description, its images, notes and statistics."""
    return dataset_directory / WORKBENCH_FOLDER


def copy_workbench_folder(source: Path, destination: Path, skip: set[str]) -> None:
    """Copy the nnWorkbench folder into a new annotation round, without the files named in ``skip`` (statistics
    that no longer match the new labels)."""
    if workbench_folder(source).is_dir():
        shutil.copytree(workbench_folder(source), workbench_folder(destination), dirs_exist_ok=True,
                        ignore=lambda _folder, names: [name for name in names if name in skip])


def trained_models(dataset_path: Path | None) -> list[Path]:
    """Return result directories that contain one or more nnU-Net folds."""
    if dataset_path is None or not dataset_path.is_dir():
        return []
    models = []
    for path in dataset_path.rglob("*"):
        try:
            if path.is_dir() and any(is_fold_folder(child) for child in path.iterdir()):
                models.append(path.relative_to(dataset_path))
        except OSError:  # unreadable folder (permissions, removed meanwhile): not a model
            continue
    return sorted(models, key=lambda path: str(path).casefold())


def model_trainer_plan_configuration(model_path: Path) -> tuple[str, str, str] | None:
    """Extract trainer, plans identifier, and configuration from a model folder name."""
    parts = model_path.name.split("__")
    return (parts[-3], parts[-2], parts[-1]) if len(parts) >= 3 else None


def model_folds(model_path: Path) -> list[Path]:
    """Return the fold directories for an nnU-Net trained model."""
    if not model_path.is_dir():
        return []
    try:
        return sorted((path for path in model_path.iterdir() if is_fold_folder(path)), key=lambda path: path.name)
    except OSError:
        return []


def json_object(path: Path) -> dict | None:
    """Read a JSON object from a file, returning ``None`` when unavailable."""
    data = json_value(path)
    return data if isinstance(data, dict) else None


def json_value(path: Path) -> object | None:
    """Read any valid JSON value from a file, returning ``None`` when unavailable."""
    try:
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    # ValueError includes JSONDecodeError and UnicodeDecodeError (e.g. UTF-16 from Notepad or PowerShell);
    # RecursionError: deeply nested arrays
    except (OSError, ValueError, RecursionError):
        return None
    return data
