from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from .dataset_contents import copy_workbench_folder
from .dataset_json import listed_cases
from .dataset_scanner import id_in_use
from .image_files import image_suffix
from .label_analysis import ANALYSIS_FILENAME

MAX_NAME_LENGTH = 200  # folder names may have at most 255 bytes, "Dataset123456_" included


def normalise_name(name: str) -> str:
    """Letters and digits only (A-Z, 0-9), everything else becomes "_": "Leber CT" -> "Leber_CT"."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    if not cleaned:
        raise ValueError("Dataset name must contain letters or numbers")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise ValueError(f"Dataset name is too long (at most {MAX_NAME_LENGTH} characters)")
    return cleaned


def directory_name(dataset_id: int, name: str) -> str:
    return f"Dataset{dataset_id:03d}_{normalise_name(name)}"


def create_dataset(raw_root: Path | None, all_roots: list[Path | None], dataset_id: int,
                   name: str) -> Path:
    if raw_root is None or not raw_root.is_dir():
        raise ValueError("Choose an existing nnUNet_raw directory before creating a dataset")
    if dataset_id <= 0:
        raise ValueError("Dataset ID must be a positive integer")
    if id_in_use(dataset_id, all_roots):
        raise ValueError(f"Dataset ID {dataset_id} already exists in a configured directory")
    folder = directory_name(dataset_id, name)
    destination = raw_root / folder
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise ValueError(f"{folder} already exists") from error
    try:
        for child in ("imagesTr", "labelsTr"):
            (destination / child).mkdir()
        labels = {"background": 0, "": 1}
        manifest = {"channel_names": {"0": ""}, "labels": labels, "numTraining": 0,
                    "file_ending": "", "name": folder, "description": "", "licence": ""}
        (destination / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)  # no half-created dataset
        raise
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


INHERITS_FROM = "inherits_from"  # dataset.json key of a new iteration: the folder name of the dataset it came from


def create_next_iteration(raw_root: Path | None, all_roots: list[Path | None], source: Path,
                          dataset_id: int, name: str, *, link_images: bool = False) -> Path:
    """Create a raw annotation-workflow iteration from its images, labels and manifest; its dataset.json names the
    source (``inherits_from``).

    Copy (default): the images go into imagesTr; a source whose dataset.json lists its files ("dataset" key) gets
    them copied into imagesTr/labelsTr, named as nnU-Net expects (case_0000…, case), and a dataset.json without the
    list. ``link_images``: the images stay where they are and dataset.json lists them (relative paths, so moving
    nnUNet_raw as a whole keeps them working); the labels are copied into labelsTr, they change in the new iteration."""
    if raw_root is None or not raw_root.is_dir():
        raise ValueError("Choose an existing nnUNet_raw directory before creating an iteration")
    listed = listed_cases(source)
    if not source.is_dir() or not (source / "dataset.json").is_file() or (
            listed is None and not (source / "imagesTr").is_dir()):
        raise ValueError("The selected dataset must contain imagesTr and dataset.json")
    if dataset_id <= 0:
        raise ValueError("Dataset ID must be a positive integer")
    if id_in_use(dataset_id, all_roots):
        raise ValueError(f"Dataset ID {dataset_id} already exists in a configured directory")
    destination = raw_root / directory_name(dataset_id, name)
    if destination.exists():
        raise ValueError(f"{destination.name} already exists")
    try:
        destination.mkdir()
        if link_images:
            _link_images(source, destination)
        elif listed is not None:
            _copy_listed(source, destination, listed)
        else:
            shutil.copytree(source / "imagesTr", destination / "imagesTr")
            labels = source / "labelsTr"
            if labels.is_dir():
                shutil.copytree(labels, destination / "labelsTr")
            else:
                (destination / "labelsTr").mkdir()
            shutil.copy2(source / "dataset.json", destination / "dataset.json")
        manifest = json.loads((destination / "dataset.json").read_text(encoding="utf-8"))
        manifest[INHERITS_FROM] = source.name
        _write_manifest(destination, manifest)
        # description, its images and the notes stay with every iteration; the statistics belong to the old labels
        copy_workbench_folder(source, destination, skip={ANALYSIS_FILENAME})
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def _copy_listed(source: Path, destination: Path, listed: dict[str, tuple[tuple[Path, ...], Path]]) -> None:
    """Copy the files of a "dataset" file list into imagesTr/labelsTr (missing ones are skipped) and write
    dataset.json without the list, so nnU-Net reads the folders of the new iteration."""
    manifest = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    ending = manifest.get("file_ending") if isinstance(manifest.get("file_ending"), str) else ""
    (destination / "imagesTr").mkdir()
    (destination / "labelsTr").mkdir()
    for case, (images, label) in listed.items():
        for index, image in enumerate(images):
            if image.is_file():
                shutil.copy2(image, destination / "imagesTr" / f"{case}_{index:04d}{ending or image_suffix(image)}")
        if label.is_file():
            shutil.copy2(label, destination / "labelsTr" / f"{case}{ending or image_suffix(label)}")
    manifest.pop("dataset", None)
    _write_manifest(destination, manifest)


def _link_images(source: Path, destination: Path) -> None:
    """The new iteration lists the source's images in dataset.json ("dataset") instead of copying them; its labels are
    copies in its own labelsTr (cases without a label list the file Save annotation will write)."""
    from .workflow_status import raw_overview

    manifest = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    ending = manifest.get("file_ending") if isinstance(manifest.get("file_ending"), str) else ""
    (destination / "imagesTr").mkdir()
    (destination / "labelsTr").mkdir()
    listing = {}
    for case in raw_overview(source, manifest).cases:  # imagesTr or the source's own file list
        if not case.image_files:
            continue
        label = f"{case.case_id}{ending or image_suffix(case.label_file or case.image_files[0])}"
        if case.label_file is not None:
            shutil.copy2(case.label_file, destination / "labelsTr" / label)
        listing[case.case_id] = {"images": [_listed_path(path, destination) for path in case.image_files],
                                 "label": f"labelsTr/{label}"}
    manifest["dataset"] = listing
    _write_manifest(destination, manifest)


def _listed_path(path: Path, dataset: Path) -> str:
    """A file for dataset.json's file list: relative to the dataset folder, with / (nnU-Net may run on Linux)."""
    try:
        return Path(os.path.relpath(path, dataset)).as_posix()
    except ValueError:  # another drive on Windows
        return str(path)


def _write_manifest(dataset: Path, manifest: dict) -> None:
    (dataset / "dataset.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
