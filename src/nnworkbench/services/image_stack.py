"""Stack ordered 2D image slices into one MITK-written 3D NIfTI volume."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

_NATURAL_PARTS = re.compile(r"(\d+)")


def image_files(input_directory: Path | str, file_type: str) -> list[Path]:
    """Return matching image files in natural filename order.

    MITK determines whether each matching file is readable; every image must be 2D.
    """
    directory = Path(input_directory)
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    suffix = file_type if file_type.startswith(".") else f".{file_type}"
    paths = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() == suffix.casefold()
    ]
    if not paths:
        raise ValueError(f"No {suffix} files were found in {directory}.")
    return sorted(paths, key=lambda path: tuple(
        int(part) if part.isdigit() else part.casefold() for part in _NATURAL_PARTS.split(path.name)
    ))


def default_output_path(input_directory: Path | str, scale_factor: int = 1) -> Path:
    """Return the default NIfTI path beside the input directory.

    A non-default scale factor is included so a downsampled volume cannot be
    mistaken for the full-resolution default output.
    """
    directory = Path(input_directory)
    suffix = "" if scale_factor == 1 else f"_scale{scale_factor}"
    return directory.parent / f"{directory.name}{suffix}.nii.gz"


@contextmanager
def suppress_native_output(enabled: bool) -> Iterator[None]:
    """Temporarily silence MITK/ITK messages emitted directly by native code."""
    if not enabled:
        yield
        return
    try:
        saved_stdout = os.dup(1)
    except OSError:  # no standard output (e.g. a windowed process): nothing to silence
        yield
        return
    try:
        saved_stderr = os.dup(2)
    except OSError:
        os.close(saved_stdout)
        yield
        return
    with open(os.devnull, "wb") as null:
        try:
            os.dup2(null.fileno(), 1); os.dup2(null.fileno(), 2)
            yield
        finally:
            os.dup2(saved_stdout, 1); os.dup2(saved_stderr, 2)
            os.close(saved_stdout); os.close(saved_stderr)


def _read_image_slice(path: Path):
    import mitk
    import numpy as np

    array = np.asarray(mitk.Image(path)).copy()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image slice, but {path.name} has shape {array.shape}.")
    return array


def _load_stack(files: Sequence[Path], workers: int, quiet: bool):
    import numpy as np

    with suppress_native_output(quiet):
        first = _read_image_slice(files[0])
        volume = np.empty((len(files), *first.shape), dtype=first.dtype)
        volume[0] = first
        if len(files) == 1:
            return volume
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-stack") as executor:
            for index, slice_data in enumerate(executor.map(_read_image_slice, files[1:]), start=1):
                if slice_data.shape != first.shape:
                    raise ValueError(
                        f"Slice {files[index].name} has shape {slice_data.shape}; "
                        f"expected {first.shape}."
                    )
                if slice_data.dtype != first.dtype:
                    raise ValueError(
                        f"Slice {files[index].name} has dtype {slice_data.dtype}; "
                        f"expected {first.dtype}."
                    )
                volume[index] = slice_data
    return volume


def _write_volume(volume, output_path: Path, spacing: tuple[float, float, float]) -> None:
    import mitk

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mitk.Image(volume, spacing=spacing).save(output_path)


def _validated_spacing(spacing: Sequence[float] | None) -> tuple[float, float, float]:
    values = (1.0, 1.0, 1.0) if spacing is None else tuple(float(value) for value in spacing)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("Spacing must contain three positive values in X Y Z order.")
    return values  # type: ignore[return-value]


def stack_image_folder(
    input_directory: Path | str, *, file_type: str, output_path: Path | str | None = None,
    workers: int | None = None, scale_factor: int = 1, spacing: Sequence[float] | None = None,
    overwrite: bool = False, quiet: bool = True, progress: Callable[[str], None] | None = None,
) -> Path:
    """Create a 3D NIfTI volume from naturally ordered 2D image slices.

    ``file_type`` is a filename extension supported by MITK, such as ``.tif``,
    ``.tiff`` or ``.png``. All matching files must contain 2D images with equal shape
    and data type. ``spacing`` is given as X, Y, Z. Integer ``scale_factor``
    downsamples all three volume axes. A supplied spacing is increased by that
    factor; omitted spacing remains the unit spacing ``(1, 1, 1)``.
    """
    if scale_factor < 1:
        raise ValueError("Scale factor must be a positive integer.")
    files = image_files(input_directory, file_type)
    requested_workers = workers if workers is not None else min(4, os.cpu_count() or 1)
    if requested_workers < 1:
        raise ValueError("Workers must be at least 1.")
    worker_count = min(requested_workers, len(files))
    destination = (
        Path(output_path) if output_path is not None else default_output_path(input_directory, scale_factor)
    )
    if destination.exists() and not overwrite:
        raise ValueError(f"Output already exists: {destination}. Use overwrite=True to replace it.")
    if progress is not None:
        progress(
            f"Loading {len(files)} {files[0].suffix} slice(s) with {worker_count} worker(s)…"
        )
    volume = _load_stack(files, worker_count, quiet)
    if scale_factor > 1:
        volume = volume[::scale_factor, ::scale_factor, ::scale_factor]
    written_spacing = _validated_spacing(spacing)
    if spacing is not None:
        written_spacing = tuple(value * scale_factor for value in written_spacing)
    _write_volume(volume, destination, written_spacing)
    if progress is not None:
        progress(
            f"Wrote {destination} with shape {tuple(reversed(volume.shape))} "
            f"and spacing {written_spacing}."
        )
    return destination
