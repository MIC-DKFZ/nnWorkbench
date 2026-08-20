from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .dataset_scanner import id_in_use


def normalise_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    if not cleaned:
        raise ValueError("Dataset name must contain letters or numbers")
    return cleaned


def create_dataset(raw_root: Path | None, all_roots: list[Path | None], dataset_id: int,
                   name: str, include_annotations: bool = False) -> Path:
    if raw_root is None or not raw_root.is_dir():
        raise ValueError("Choose an existing nnUNet_raw directory before creating a dataset")
    if dataset_id <= 0:
        raise ValueError("Dataset ID must be a positive integer")
    if id_in_use(dataset_id, all_roots):
        raise ValueError(f"Dataset ID {dataset_id} already exists in a configured directory")
    directory_name = f"Dataset{dataset_id:03d}_{normalise_name(name)}"
    destination = raw_root / directory_name
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise ValueError(f"{directory_name} already exists") from error
    for child in ("imagesTr", "labelsTr", "imagesTs", *( ("annotationsTr",) if include_annotations else ())):
        (destination / child).mkdir()
    labels = {"background": 0, "": 1}
    manifest = {"channel_names": {"0": ""}, "labels": labels, "numTraining": 0,
                "file_ending": "", "name": directory_name, "description": "", "license": ""}
    (destination / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def next_iteration_name(name: str) -> str:
    """Return the next ``_iterN`` variant of a dataset name."""
    match = re.match(r"^(?P<base>.+)_iter(?P<number>\d+)$", name, flags=re.IGNORECASE)
    if match:
        return f"{match['base']}_iter{int(match['number']) + 1}"
    return f"{name}_iter1"


def next_free_dataset_id(start: int, all_roots: list[Path | None]) -> int:
    """Return the first unused dataset ID at or after ``start`` across configured roots."""
    candidate = max(1, start)
    while id_in_use(candidate, all_roots):
        candidate += 1
    return candidate


def create_next_iteration(raw_root: Path | None, all_roots: list[Path | None], source: Path,
                          dataset_id: int, name: str) -> Path:
    """Create a raw annotation-workflow iteration from its images, annotations and manifest."""
    if raw_root is None or not raw_root.is_dir():
        raise ValueError("Choose an existing nnUNet_raw directory before creating an iteration")
    if not source.is_dir() or not (source / "imagesTr").is_dir() or not (source / "dataset.json").is_file():
        raise ValueError("The selected dataset must contain imagesTr and dataset.json")
    if dataset_id <= 0:
        raise ValueError("Dataset ID must be a positive integer")
    if id_in_use(dataset_id, all_roots):
        raise ValueError(f"Dataset ID {dataset_id} already exists in a configured directory")
    destination = raw_root / f"Dataset{dataset_id:03d}_{normalise_name(name)}"
    if destination.exists():
        raise ValueError(f"{destination.name} already exists")
    try:
        destination.mkdir()
        shutil.copytree(source / "imagesTr", destination / "imagesTr")
        annotations = source / "annotationsTr"
        if annotations.is_dir():
            shutil.copytree(annotations, destination / "annotationsTr")
        else:
            (destination / "annotationsTr").mkdir()
        shutil.copy2(source / "dataset.json", destination / "dataset.json")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination
