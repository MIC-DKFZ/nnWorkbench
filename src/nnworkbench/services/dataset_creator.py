from __future__ import annotations

import json
import re
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
