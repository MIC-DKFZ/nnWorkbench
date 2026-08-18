from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    for key in ("data_root", "dataset_id", "modalities", "labels"):
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    return config


def dataset_dir(config: dict, base_dir: Path) -> Path:
    return base_dir / config["data_root"] / "nnUNet_raw" / config["dataset_id"]


def initialise(config: dict, base_dir: Path) -> Path:
    root = dataset_dir(config, base_dir)
    for folder in ("imagesTr", "labelsTr", "imagesTs"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for folder in ("scribbles", "drafts", "approved", "reviews"):
        (base_dir / config["data_root"] / "work" / folder).mkdir(parents=True, exist_ok=True)
    manifest = root / "dataset.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                {"channel_names": config["modalities"], "labels": config["labels"], "numTraining": 0,
                 "file_ending": ".nii.gz"}, indent=2
            ) + "\n",
            encoding="utf-8",
        )
    return root


def case_ids(images_dir: Path) -> set[str]:
    return {path.name.rsplit("_", 1)[0] for path in images_dir.glob("*_0000.nii.gz")}


def status(config: dict, base_dir: Path) -> dict[str, int]:
    root = dataset_dir(config, base_dir)
    images = case_ids(root / "imagesTr")
    labels = {path.name.removesuffix(".nii.gz") for path in (root / "labelsTr").glob("*.nii.gz")}
    return {"images": len(images), "accepted_labels": len(labels), "pending": len(images - labels)}


def accept_label(config: dict, base_dir: Path, case_id: str, source: Path, replace: bool = False) -> Path:
    root = dataset_dir(config, base_dir)
    if case_id not in case_ids(root / "imagesTr"):
        raise ValueError(f"No training image found for case '{case_id}'")
    if not source.is_file() or not source.name.endswith(".nii.gz"):
        raise ValueError("Approved label must be an existing .nii.gz file")
    destination = root / "labelsTr" / f"{case_id}.nii.gz"
    if destination.exists() and not replace:
        raise FileExistsError(f"{destination} already exists; use --replace after review")
    shutil.copy2(source, destination)
    return destination
