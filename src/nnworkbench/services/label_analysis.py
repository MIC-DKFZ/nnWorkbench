"""Background-friendly label-distribution analysis for raw nnU-Net datasets."""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .dataset_json import read_manifest

ANALYSIS_FILENAME = "class_distribution.json"


def analysis_path(dataset_directory: Path) -> Path:
    """Return the on-disk cache location for a raw dataset."""
    return dataset_directory / ANALYSIS_FILENAME


def label_files(dataset_directory: Path) -> list[Path]:
    """Return NIfTI training label files, in a stable order."""
    labels_directory = dataset_directory / "labelsTr"
    if not labels_directory.is_dir():
        return []
    return sorted((path for path in labels_directory.iterdir()
                   if path.is_file() and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))),
                  key=lambda path: path.name.casefold())


def _analyse_file(filename: str) -> dict[int, int]:
    """Return voxel counts for one NIfTI label map (safe to run in a child process)."""
    import mitk
    import numpy as np

    image = mitk.Image(filename)
    values, counts = np.unique(np.asarray(image), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


def _label_names(dataset_directory: Path) -> dict[int, str]:
    try:
        labels = read_manifest(dataset_directory).get("labels", {})
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(labels, dict):
        return {}
    names: dict[int, str] = {}
    for name, value in labels.items():
        if isinstance(value, int):
            names.setdefault(value, str(name))
    return names


def _ignored_label_ids(dataset_directory: Path) -> set[int]:
    """Return IDs explicitly named ``ignore`` in the raw dataset manifest."""
    try:
        labels = read_manifest(dataset_directory).get("labels", {})
    except (OSError, ValueError, json.JSONDecodeError):
        return set()
    ignore = labels.get("ignore") if isinstance(labels, dict) else None
    return {ignore} if isinstance(ignore, int) else set()


def aggregate_class_counts(per_file: Iterable[dict[int, int]], label_names: dict[int, str],
                           ignored_label_ids: set[int] | None = None) -> list[dict[str, int | str]]:
    """Combine per-file histograms into display-ready per-class statistics."""
    ignored = ignored_label_ids or set()
    images = Counter()
    voxels = Counter()
    for counts in per_file:
        for class_id, count in counts.items():
            if class_id not in ignored and count > 0:
                images[class_id] += 1
                voxels[class_id] += count
    class_ids = sorted((set(label_names) | set(voxels)) - ignored)
    return [{"id": class_id, "name": label_names.get(class_id, f"Class {class_id}"),
             "images": images[class_id], "voxels": voxels[class_id]} for class_id in class_ids]


def undefined_label_ids(files: Iterable[Path], per_file: Iterable[dict[int, int]],
                        defined_ids: set[int]) -> dict[str, list[int]]:
    """Return undefined label IDs grouped by their affected label-map filename."""
    return {
        path.name: sorted(set(counts) - defined_ids)
        for path, counts in zip(files, per_file, strict=True)
        if set(counts) - defined_ids
    }


def analyse_label_distribution(dataset_directory: Path, workers: int | None = None) -> dict:
    """Analyse all ``labelsTr`` files and atomically cache the result in the raw folder.

    The work is spread across separate processes so image loading and voxel counting do
    not occupy the GUI thread. ``workers`` is capped at the number of label files.
    """
    files = label_files(dataset_directory)
    if not files:
        raise ValueError("No NIfTI label files were found in labelsTr.")
    available = os.cpu_count() or 1
    worker_count = max(1, min(workers or available, available, len(files)))
    # Spawn avoids forking a running Qt process from its background worker thread.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        counts = list(executor.map(_analyse_file, (str(path) for path in files)))
    label_names = _label_names(dataset_directory)
    unexpected = undefined_label_ids(files, counts, set(label_names))
    ignored = _ignored_label_ids(dataset_directory)
    per_image = [
        {"image": path.name,
         "voxels": {str(class_id): count for class_id, count in file_counts.items() if class_id not in ignored}}
        for path, file_counts in zip(files, counts, strict=True)
    ]
    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels_directory": "labelsTr",
        "files_analyzed": len(files),
        "workers": worker_count,
        "classes": aggregate_class_counts(
            counts, label_names, ignored
        ),
        "per_image": per_image,
        "undefined_label_ids": unexpected,
    }
    _write_analysis(analysis_path(dataset_directory), analysis)
    return analysis


def load_label_analysis(dataset_directory: Path) -> dict | None:
    """Read a previously cached label analysis, if it is valid."""
    try:
        with analysis_path(dataset_directory).open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("classes"), list) else None


def _write_analysis(target: Path, data: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".class-distribution.", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
