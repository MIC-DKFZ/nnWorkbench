from __future__ import annotations

import json
from pathlib import Path


RAW_FOLDERS = ("imagesTr", "labelsTr", "annotationsTr")


def raw_files(dataset_path: Path | None) -> dict[str, list[Path]]:
    """Return the files available in the standard raw-data folders."""
    if dataset_path is None:
        return {}
    contents: dict[str, list[Path]] = {}
    for folder_name in RAW_FOLDERS:
        folder = dataset_path / folder_name
        if folder.is_dir():
            contents[folder_name] = sorted(path for path in folder.iterdir() if path.is_file())
    return contents


def plan_files(dataset_path: Path | None) -> list[Path]:
    """Return plan JSON files relative to a preprocessed dataset directory."""
    return [path for path in preprocessed_files(dataset_path) if "plan" in path.name.casefold()]


def preprocessed_files(dataset_path: Path | None) -> list[Path]:
    """Return every file relative to a preprocessed dataset directory."""
    if dataset_path is None or not dataset_path.is_dir():
        return []
    return sorted((path.relative_to(dataset_path) for path in dataset_path.rglob("*") if path.is_file()),
                  key=lambda path: str(path).casefold())


def trained_models(dataset_path: Path | None) -> list[Path]:
    """Return result directories that contain one or more nnU-Net folds."""
    if dataset_path is None or not dataset_path.is_dir():
        return []
    models = []
    for path in dataset_path.rglob("*"):
        if path.is_dir() and any(child.is_dir() and child.name.startswith("fold_") for child in path.iterdir()):
            models.append(path.relative_to(dataset_path))
    return sorted(models, key=lambda path: str(path).casefold())


def model_plan_configuration(model_path: Path) -> tuple[str, str] | None:
    """Extract the plans identifier and configuration from an nnU-Net model folder name."""
    parts = model_path.name.split("__")
    return (parts[-2], parts[-1]) if len(parts) >= 3 else None


def model_folds(model_path: Path) -> list[Path]:
    """Return the fold directories for an nnU-Net trained model."""
    if not model_path.is_dir():
        return []
    return sorted((path for path in model_path.iterdir()
                   if path.is_dir() and path.name.startswith("fold_")), key=lambda path: path.name)


def validation_files(fold_path: Path) -> list[Path]:
    """Return files below a fold's validation directory, relative to that directory."""
    directory = fold_path / "validation"
    if not directory.is_dir():
        return []
    return sorted((path.relative_to(directory) for path in directory.rglob("*") if path.is_file()),
                  key=lambda path: str(path).casefold())


def validation_summary(fold_path: Path) -> dict | None:
    """Read a fold validation summary, if one exists and is a JSON object."""
    return json_object(fold_path / "validation" / "summary.json")


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
