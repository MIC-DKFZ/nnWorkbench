"""Background-friendly label-distribution analysis for raw nnU-Net datasets."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from .dataset_contents import workbench_folder
from .dataset_json import read_manifest
from .image_files import has_image_suffix, strip_image_suffix

ANALYSIS_FILENAME = "class_distribution.json"
_CHANNEL_SUFFIX = re.compile(r"_\d{4}$")


def analysis_path(dataset_directory: Path) -> Path:
    """Return the on-disk cache location for a raw dataset (in its nnworkbench folder)."""
    return workbench_folder(dataset_directory) / ANALYSIS_FILENAME


def label_files(dataset_directory: Path) -> list[Path]:
    """Return training label images, in a stable order."""
    labels_directory = dataset_directory / "labelsTr"
    if not labels_directory.is_dir():
        return []
    return sorted((path for path in labels_directory.iterdir()
                   if path.is_file() and has_image_suffix(path)),
                  key=lambda path: path.name.casefold())


def _case_id(path: Path) -> str:
    """Return an nnU-Net case ID from an image or label filename."""
    return _CHANNEL_SUFFIX.sub("", strip_image_suffix(path))


def _image_case_ids(dataset_directory: Path) -> set[str]:
    images_directory = dataset_directory / "imagesTr"
    if not images_directory.is_dir():
        return set()
    return {
        _case_id(path) for path in images_directory.iterdir()
        if path.is_file() and has_image_suffix(path)
    }


def label_validity(counts: dict[int, int], filename: Path, image_cases: set[str],
                   background_ids: set[int], ignored_ids: set[int], undefined_ids: set[int]) -> list[str]:
    """Return all per-label findings shown in the statistics details table."""
    findings: list[str] = []
    if set(counts).issubset(background_ids | ignored_ids):
        findings.append("Empty (background/ignore only)")
    if _case_id(filename) not in image_cases:
        findings.append("No matching imagesTr file")
    if undefined_ids:
        findings.append(f"Undefined label IDs: {', '.join(str(value) for value in sorted(undefined_ids))}")
    return findings


def _analyse_file(filename: str) -> dict:
    """Return label counts and spatial metadata for one map in a child process."""
    import mitk
    import numpy as np

    image = mitk.Image(filename)
    array = np.asarray(image)
    values, counts = np.unique(array, return_counts=True)
    dimensions = min(len(image.shape), 3)
    spacing = image.spacing
    return {
        "voxels": {int(value): int(count) for value, count in zip(values, counts, strict=True)},
        # MITK spacing is X/Y/Z, while Image.shape follows NumPy's Z/Y/X order.
        "shape": [int(value) for value in reversed(image.shape[-dimensions:])],
        "spacing": [float(spacing[index]) for index in range(dimensions)],
    }


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


def analyse_label_distribution(dataset_directory: Path, workers: int | None = None,
                               progress: Callable[[str], None] | None = None) -> dict:
    """Analyse all ``labelsTr`` files and atomically cache the result in the raw folder.

    The work is spread across separate processes so image loading and voxel counting do
    not occupy the GUI thread. ``workers`` is capped at the number of label files.
    """
    files = label_files(dataset_directory)
    if not files:
        raise ValueError("No label images were found in labelsTr.")
    available = os.cpu_count() or 1
    worker_count = max(1, min(workers or available, available, len(files)))
    # Spawn avoids forking a running Qt process from its background worker thread.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        file_analyses = []
        for index, (path, file_analysis) in enumerate(
                zip(files, executor.map(_analyse_file, (str(path) for path in files)), strict=True), start=1):
            file_analyses.append(file_analysis)
            if progress is not None:
                progress(f"Analyzed label {index}/{len(files)}: {path.name}")
    counts = [file_analysis["voxels"] for file_analysis in file_analyses]
    label_names = _label_names(dataset_directory)
    unexpected = undefined_label_ids(files, counts, set(label_names))
    ignored = _ignored_label_ids(dataset_directory)
    background = {class_id for class_id, name in label_names.items() if name.casefold() == "background"} or {0}
    image_cases = _image_case_ids(dataset_directory)
    per_image = [
        {
            "image": path.name,
            "voxels": {
                str(class_id): count for class_id, count in file_analysis["voxels"].items()
                if class_id not in ignored
            },
            "shape": file_analysis["shape"],
            "spacing": file_analysis["spacing"],
            "validity": label_validity(
                file_analysis["voxels"], path, image_cases, background, ignored,
                set(unexpected.get(path.name, [])),
            ),
        }
        for path, file_analysis in zip(files, file_analyses, strict=True)
    ]
    analysis = {
        "generated_at": datetime.now(UTC).isoformat(),
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
    """Read a current-format cached label analysis."""
    try:
        with analysis_path(dataset_directory).open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("classes"), list):
        return None
    per_image = data.get("per_image")
    if not isinstance(per_image, list):
        return None
    required_fields = {"image", "voxels", "shape", "spacing", "validity"}
    return data if all(isinstance(item, dict) and required_fields <= item.keys() for item in per_image) else None


def _write_analysis(target: Path, data: dict) -> None:
    target.parent.mkdir(exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".class-distribution.", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    """Run label analysis as a console-reporting subprocess."""
    parser = argparse.ArgumentParser(description="Analyze nnU-Net training labels.")
    parser.add_argument("dataset_directory", type=Path, help="Raw dataset directory")
    parser.add_argument("--workers", type=int, default=None, help="Maximum worker processes")
    arguments = parser.parse_args()
    try:
        files = label_files(arguments.dataset_directory)
        print(f"Computing statistics of {len(files)} label file(s) in {arguments.dataset_directory}", flush=True)
        analysis = analyse_label_distribution(
            arguments.dataset_directory, arguments.workers,
            progress=lambda message: print(message, flush=True),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Computing statistics failed: {error}") from error
    print(f"Statistics computed: {analysis_path(arguments.dataset_directory)} "
          f"({analysis['files_analyzed']} file(s))", flush=True)


if __name__ == "__main__":
    main()
