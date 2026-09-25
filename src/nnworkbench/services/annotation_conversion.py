"""Convert nnInteractive semantic annotations into nnU-Net training labels."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .dataset_json import read_manifest
from .image_files import has_image_suffix, image_suffix


@dataclass(frozen=True)
class AnnotationConversionResult:
    """Outcome of one annotation-to-label conversion batch."""

    converted: tuple[Path, ...]
    warnings: tuple[str, ...]
    workers: int


def annotation_files(dataset_directory: Path) -> list[Path]:
    """Return semantic annotation images in stable filename order."""
    directory = dataset_directory / "annotationsTr"
    if not directory.is_dir():
        return []
    return sorted((path for path in directory.iterdir() if path.is_file() and has_image_suffix(path)),
                  key=lambda path: path.name.casefold())


def has_annotation_files(dataset_directory: Path) -> bool:
    """Return whether annotationsTr contains at least one convertible annotation image."""
    directory = dataset_directory / "annotationsTr"
    return directory.is_dir() and any(
        path.is_file() and has_image_suffix(path) for path in directory.iterdir()
    )


def _ignore_label_id(dataset_directory: Path) -> int:
    """Read and validate nnU-Net's final ignore label from the raw manifest."""
    manifest_path = dataset_directory / "dataset.json"
    if not manifest_path.is_file():
        raise ValueError("A raw dataset.json is required before converting annotations.")
    manifest = read_manifest(dataset_directory)
    labels = manifest.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("The raw dataset.json must contain a labels object.")
    ignore = labels.get("ignore")
    values = [value for value in labels.values() if isinstance(value, int)]
    if not isinstance(ignore, int) or not values:
        raise ValueError("Add an integer 'ignore' label to the raw dataset.json before converting annotations.")
    if ignore != max(values):
        raise ValueError("The 'ignore' label in the raw dataset.json must use the last (highest) label ID.")
    return ignore


def _temporary_target(target: Path) -> Path:
    suffix = image_suffix(target) or target.suffix
    stem = target.name[:-len(suffix)]
    return target.with_name(f".{stem}.conversion{suffix}")


def _convert_file(arguments: tuple[str, str, int]) -> tuple[str | None, str | None]:
    """Convert one file in a worker process, returning a target or a warning."""
    import mitk
    import numpy as np

    annotation_name, labels_directory, ignore_id = arguments
    annotation_path = Path(annotation_name)
    try:
        source = mitk.Image(annotation_path)
        data = np.asarray(source)
        values = np.unique(data)
        if not np.any(data != 0):
            return None, f"{annotation_path.name}: skipped because the annotation is empty."
        invalid = values[(values < 0) | (values > ignore_id)]
        if invalid.size:
            shown = ", ".join(str(int(value)) for value in invalid[:8])
            suffix = "…" if invalid.size > 8 else ""
            return None, (
                f"{annotation_path.name}: skipped because it contains out-of-range label value(s): {shown}{suffix}."
            )
        output = np.where(data == 0, ignore_id, data - 1).astype(np.uint16, copy=False)
        target = Path(labels_directory) / annotation_path.name
        temporary = _temporary_target(target)
        try:
            mitk.Image(output, spacing=source.spacing, origin=source.origin, direction=source.direction).save(temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    except Exception as error:
        return None, f"{annotation_path.name}: skipped because conversion failed: {error}"
    return str(target), None


def convert_annotations(dataset_directory: Path, workers: int | None = None,
                        progress: Callable[[str], None] | None = None) -> AnnotationConversionResult:
    """Convert semantic maps to labelsTr, preserving geometry and skipping bad cases.

    Annotation value ``0`` is nnInteractive's ignore value and becomes the final
    nnU-Net ignore ID. Every positive value is shifted down by one. Values outside
    ``0..ignore_id`` and all-zero annotations are skipped with an individual warning.
    """
    ignore_id = _ignore_label_id(dataset_directory)
    files = annotation_files(dataset_directory)
    targets = dataset_directory / "labelsTr"
    targets.mkdir(parents=True, exist_ok=True)
    available = os.cpu_count() or 1
    worker_count = max(1, min(workers or available, available, len(files))) if files else 1
    converted: list[Path] = []
    warnings: list[str] = []
    # Spawn avoids forking a running Qt process from the GUI's worker thread.
    context = multiprocessing.get_context("spawn")
    arguments = ((str(path), str(targets), ignore_id) for path in files)
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        for index, (target, warning) in enumerate(executor.map(_convert_file, arguments), start=1):
            if target:
                converted.append(Path(target))
                if progress is not None:
                    progress(f"Converted annotation {index}/{len(files)}: {Path(target).name}")
            if warning:
                warnings.append(warning)
                if progress is not None:
                    progress(f"Warning {index}/{len(files)}: {warning}")
    return AnnotationConversionResult(tuple(converted), tuple(warnings), worker_count)
