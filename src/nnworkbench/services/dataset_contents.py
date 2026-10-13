from __future__ import annotations

import json
from pathlib import Path


def trained_models(dataset_path: Path | None) -> list[Path]:
    """Return result directories that contain one or more nnU-Net folds."""
    if dataset_path is None or not dataset_path.is_dir():
        return []
    models = []
    for path in dataset_path.rglob("*"):
        if path.is_dir() and any(child.is_dir() and child.name.startswith("fold_") for child in path.iterdir()):
            models.append(path.relative_to(dataset_path))
    return sorted(models, key=lambda path: str(path).casefold())


def model_trainer_plan_configuration(model_path: Path) -> tuple[str, str, str] | None:
    """Extract trainer, plans identifier, and configuration from a model folder name."""
    parts = model_path.name.split("__")
    return (parts[-3], parts[-2], parts[-1]) if len(parts) >= 3 else None


def model_folds(model_path: Path) -> list[Path]:
    """Return the fold directories for an nnU-Net trained model."""
    if not model_path.is_dir():
        return []
    return sorted((path for path in model_path.iterdir()
                   if path.is_dir() and path.name.startswith("fold_")), key=lambda path: path.name)


def json_object(path: Path) -> dict | None:
    """Read a JSON object from a file, returning ``None`` when unavailable."""
    data = json_value(path)
    return data if isinstance(data, dict) else None


def json_value(path: Path) -> object | None:
    """Read any valid JSON value from a file, returning ``None`` when unavailable."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return data
