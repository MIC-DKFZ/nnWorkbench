"""Background-friendly label-distribution analysis for raw nnU-Net datasets."""

from __future__ import annotations

import argparse
import json
import math
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
from .dataset_json import listed_cases, read_manifest
from .image_files import has_image_suffix, strip_image_suffix

ANALYSIS_FILENAME = "class_distribution.json"
_CHANNEL_SUFFIX = re.compile(r"_\d{4}$")


def analysis_path(dataset_directory: Path) -> Path:
    """Return the on-disk cache location for a raw dataset (in its nnworkbench folder)."""
    return workbench_folder(dataset_directory) / ANALYSIS_FILENAME


def label_files(dataset_directory: Path) -> list[Path]:
    """Return training label images, in a stable order (the listed ones for a "dataset" file list in dataset.json)."""
    listed = listed_cases(dataset_directory)
    if listed is not None:
        return sorted((label for _images, label in listed.values() if label.is_file()),
                      key=lambda path: path.name.casefold())
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
    """Cases with an image, named as their label files are (label_validity compares the label's case name)."""
    listed = listed_cases(dataset_directory)
    if listed is not None:
        return {_case_id(label) for images, label in listed.values() if any(image.is_file() for image in images)}
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


UNREADABLE = "Unreadable label file"
NOT_WHOLE_NUMBERS = "Values that are not whole numbers (counted without their fraction)"


def _analyse_file(filename: str) -> dict:
    """Return label counts and spatial metadata for one map in a child process; a file that cannot be read gives
    ``{"error": message}`` instead, so one broken file does not stop the others."""
    import mitk
    import numpy as np

    from .image_stack import suppress_native_output

    try:
        with suppress_native_output(True):  # MITK logs ~10 lines per file, which would bury the progress lines
            image = mitk.Image(filename)
        array = np.asarray(image)
        values, counts = np.unique(array, return_counts=True)
        voxels: Counter = Counter()
        for value, count in zip(values, counts, strict=True):
            voxels[int(value)] += int(count)  # float labels: 0.3 and 0.7 both count as 0
        dimensions = min(len(image.shape), 3)
        spacing = image.spacing
        return {
            "voxels": dict(voxels),
            # MITK spacing is X/Y/Z, while Image.shape follows NumPy's Z/Y/X order.
            "shape": [int(value) for value in reversed(image.shape[-dimensions:])],
            "spacing": [float(spacing[index]) for index in range(dimensions)],
            "whole_numbers": bool(np.all(np.mod(values, 1) == 0)) if values.dtype.kind == "f" else True,
        }
    except Exception as error:  # noqa: BLE001  (any reader error: report the file, keep the others)
        message = str(error).strip().splitlines()
        return {"error": message[-1] if message else type(error).__name__}


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


def _file_validity(file_analysis: dict, path: Path, image_cases: set[str], background_ids: set[int],
                   ignored_ids: set[int], undefined_ids: set[int]) -> list[str]:
    if "error" in file_analysis:
        return [f"{UNREADABLE}: {file_analysis['error']}"]
    findings = label_validity(file_analysis["voxels"], path, image_cases, background_ids, ignored_ids, undefined_ids)
    return findings if file_analysis.get("whole_numbers", True) else [*findings, NOT_WHOLE_NUMBERS]


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
                progress(f"Analyzed label {index}/{len(files)}: {path.name}" if "error" not in file_analysis else
                         f"Could not read label {index}/{len(files)}: {path.name}: {file_analysis['error']}")
    unreadable = [path.name for path, file_analysis in zip(files, file_analyses, strict=True) if "error" in file_analysis]
    if len(unreadable) == len(files):
        raise ValueError(f"No label file could be read ({file_analyses[0]['error']}).")
    counts = [file_analysis.get("voxels", {}) for file_analysis in file_analyses]
    label_names = _label_names(dataset_directory)
    unexpected = undefined_label_ids(files, counts, set(label_names))
    ignored = _ignored_label_ids(dataset_directory)
    background = {class_id for class_id, name in label_names.items() if name.casefold() == "background"} or {0}
    image_cases = _image_case_ids(dataset_directory)
    per_image = [
        {
            "image": path.name,
            "voxels": {
                str(class_id): count for class_id, count in file_analysis.get("voxels", {}).items()
                if class_id not in ignored
            },
            "shape": file_analysis.get("shape", []),
            "spacing": file_analysis.get("spacing", []),
            "validity": _file_validity(file_analysis, path, image_cases, background, ignored,
                                       set(unexpected.get(path.name, []))),
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
        "unreadable_files": unreadable,
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
    return data if _valid_analysis(data) else None


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_analysis(data: object) -> bool:
    """Whether a cached analysis has the fields and types the GUI reads (an edited or broken cache is ignored and
    computed again instead of failing the Data page)."""
    if not isinstance(data, dict) or not isinstance(data.get("classes"), list) or not isinstance(
            data.get("per_image"), list):
        return False
    for item in data["classes"]:
        if not (isinstance(item, dict) and isinstance(item.get("id"), int) and not isinstance(item.get("id"), bool)
                and _is_count(item.get("images", 0)) and _is_count(item.get("voxels", 0))):
            return False
    for item in data["per_image"]:
        if not (isinstance(item, dict) and isinstance(item.get("image"), str)
                and isinstance(item.get("voxels"), dict) and isinstance(item.get("shape"), list)
                and isinstance(item.get("spacing"), list) and isinstance(item.get("validity"), list)):
            return False
        if not (all(isinstance(key, str) and re.fullmatch(r"-?[0-9]+", key) and _is_count(value)
                    for key, value in item["voxels"].items())
                and all(_is_count(value) for value in item["shape"])
                and all(_is_number(value) for value in item["spacing"])):
            return False
    return True


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
    unreadable = analysis["unreadable_files"]
    print(f"Statistics computed: {analysis_path(arguments.dataset_directory)} "
          f"({analysis['files_analyzed']} file(s)"
          + (f", {len(unreadable)} unreadable: {', '.join(unreadable)}" if unreadable else "") + ")", flush=True)


if __name__ == "__main__":
    main()
