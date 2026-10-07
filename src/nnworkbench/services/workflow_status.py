"""Read-only status of a dataset for the stage pages: cases, plans, models and validation.

Everything here only lists folders and reads small JSON or log files. Image data is never
read, so the overview stays fast even for large datasets on network drives.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .dataset_contents import (
    json_object,
    model_folds,
    model_trainer_plan_configuration,
    trained_models,
)
from .image_files import has_image_suffix, image_suffix, strip_image_suffix

_CHANNEL_SUFFIX = re.compile(r"_(\d{4})$")
_EPOCH_LINE = re.compile(r"(?m)^.*?:\s*Epoch\s+(\d+)\s*$")
_PSEUDO_DICE_LINE = re.compile(r"(?m)^.*?:\s*Pseudo dice\s*\[(.*)\]\s*$")
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# A fold counts as "training" while its log changed recently; afterwards it is "stopped".
RUNNING_LOG_AGE_SECONDS = 15 * 60


# ---------------------------------------------------------------- dataset.json helpers

def ignore_label_id(manifest: dict | None) -> int | None:
    """Return the ``ignore`` label ID, or ``None`` for dense labels."""
    labels = manifest.get("labels") if isinstance(manifest, dict) else None
    if not isinstance(labels, dict):
        return None
    for name, value in labels.items():
        if isinstance(name, str) and name.strip().casefold() == "ignore" and isinstance(value, int) \
                and not isinstance(value, bool):
            return value
    return None


def label_mode(manifest: dict | None) -> str:
    """Return ``"sparse"`` when dataset.json defines an ignore label, otherwise ``"dense"``."""
    return "sparse" if ignore_label_id(manifest) is not None else "dense"


def channel_count(manifest: dict | None) -> int:
    channels = manifest.get("channel_names") if isinstance(manifest, dict) else None
    return len(channels) if isinstance(channels, dict) and channels else 1


def label_names(manifest: dict | None) -> dict[int, str]:
    """Map label IDs to names, skipping malformed entries."""
    labels = manifest.get("labels") if isinstance(manifest, dict) else None
    if not isinstance(labels, dict):
        return {}
    return {value: name for name, value in labels.items()
            if isinstance(name, str) and isinstance(value, int) and not isinstance(value, bool)}


# ---------------------------------------------------------------- raw cases

@dataclass(frozen=True)
class CaseRow:
    """One training case as seen from the folder listing."""

    case_id: str
    image_files: tuple[Path, ...]
    channels_present: bool
    label_file: Path | None
    last_edit: float | None  # modification time of the label file

    @property
    def has_label(self) -> bool:
        return self.label_file is not None


@dataclass(frozen=True)
class RawOverview:
    cases: tuple[CaseRow, ...]
    image_files: int
    label_files: int
    channels: int
    names_valid: bool
    has_images_folder: bool

    @property
    def missing_labels(self) -> int:
        return sum(not case.has_label for case in self.cases)

    @property
    def channels_complete(self) -> bool:
        return all(case.channels_present for case in self.cases)


def _image_case(path: Path) -> tuple[str, str | None]:
    """Split an imagesTr file name into case ID and channel (``"0000"``), if present."""
    stem = strip_image_suffix(path)
    match = _CHANNEL_SUFFIX.search(stem)
    return (stem[:match.start()], match.group(1)) if match else (stem, None)


def raw_overview(raw_path: Path | None, manifest: dict | None) -> RawOverview:
    """List imagesTr and labelsTr and match them into cases."""
    channels = channel_count(manifest)
    images_dir = raw_path / "imagesTr" if raw_path else None
    labels_dir = raw_path / "labelsTr" if raw_path else None
    images: dict[str, list[Path]] = {}
    names_valid = True
    image_count = 0
    if images_dir is not None and images_dir.is_dir():
        for path in images_dir.iterdir():
            if not path.is_file() or not has_image_suffix(path):
                continue
            image_count += 1
            case, channel = _image_case(path)
            if channel is None:
                names_valid = False
            images.setdefault(case, []).append(path)
    labels: dict[str, Path] = {}
    if labels_dir is not None and labels_dir.is_dir():
        for path in labels_dir.iterdir():
            if path.is_file() and has_image_suffix(path):
                labels[strip_image_suffix(path)] = path
    rows = []
    for case in sorted(images, key=str.casefold):
        files = tuple(sorted(images[case], key=lambda item: item.name))
        found_channels = {_image_case(item)[1] for item in files}
        expected = {f"{index:04d}" for index in range(channels)}
        label = labels.get(case)
        try:
            last_edit = label.stat().st_mtime if label else None
        except OSError:
            last_edit = None
        rows.append(CaseRow(case, files, expected <= found_channels, label, last_edit))
    return RawOverview(tuple(rows), image_count, len(labels), channels, names_valid,
                       images_dir is not None and images_dir.is_dir())


def newest_label_time(raw_path: Path | None) -> float | None:
    """Return the newest modification time in labelsTr (for the outdated rules)."""
    labels_dir = raw_path / "labelsTr" if raw_path else None
    if labels_dir is None or not labels_dir.is_dir():
        return None
    times = [path.stat().st_mtime for path in labels_dir.iterdir() if path.is_file() and has_image_suffix(path)]
    return max(times) if times else None


# ---------------------------------------------------------------- preprocessing

@dataclass(frozen=True)
class ConfigurationInfo:
    name: str
    patch_size: tuple
    spacing: tuple
    batch_size: int | None
    data_identifier: str
    preprocessed_cases: int  # 0 when the configuration was only planned

    @property
    def preprocessed(self) -> bool:
        return self.preprocessed_cases > 0


@dataclass(frozen=True)
class PlansInfo:
    name: str  # plans identifier, e.g. nnUNetResEncUNetLPlans
    file: Path
    planner: str
    configurations: tuple[ConfigurationInfo, ...]
    preprocessed_time: float | None
    outdated: bool


def _count_preprocessed_cases(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.iterdir()
               if path.is_file() and path.name.endswith(".b2nd") and not path.name.endswith("_seg.b2nd"))


def plans_overview(preprocessed_path: Path | None, raw_path: Path | None = None) -> tuple[PlansInfo, ...]:
    """Describe every plans file of a preprocessed dataset and its configurations."""
    if preprocessed_path is None or not preprocessed_path.is_dir():
        return ()
    newest_label = newest_label_time(raw_path)
    plans: list[PlansInfo] = []
    for file in sorted(preprocessed_path.glob("*.json"), key=lambda item: item.name.casefold()):
        content = json_object(file)
        if not content or not isinstance(content.get("configurations"), dict):
            continue
        configurations = []
        times = []
        for name, values in content["configurations"].items():
            if not isinstance(values, dict):
                continue
            identifier = values.get("data_identifier") or f"{file.stem}_{name}"
            folder = preprocessed_path / str(identifier)
            count = _count_preprocessed_cases(folder)
            if count:
                times.append(folder.stat().st_mtime)
            configurations.append(ConfigurationInfo(
                name, tuple(values.get("patch_size") or ()), tuple(values.get("spacing") or ()),
                values.get("batch_size") if isinstance(values.get("batch_size"), int) else None,
                str(identifier), count))
        preprocessed_time = max(times) if times else None
        outdated = bool(preprocessed_time and newest_label and newest_label > preprocessed_time)
        plans.append(PlansInfo(str(content.get("plans_name") or file.stem), file,
                               str(content.get("experiment_planner_used") or "—"),
                               tuple(configurations), preprocessed_time, outdated))
    return tuple(plans)


def splits_fold_count(preprocessed_path: Path | None) -> int:
    """Return the number of folds in splits_final.json (nnU-Net's default is 5)."""
    if preprocessed_path is None:
        return 5
    from .dataset_contents import json_value

    splits = json_value(preprocessed_path / "splits_final.json")
    return len(splits) if isinstance(splits, list) and splits else 5


# ---------------------------------------------------------------- training

@dataclass(frozen=True)
class FoldInfo:
    name: str  # "0", "1", …, "all"
    path: Path
    status: str  # "done", "training", "stopped", "reduced"
    epoch: int | None
    total_epochs: int | None
    pseudo_dice: float | None
    dice: float | None  # foreground mean from validation/summary.json

    @property
    def finished(self) -> bool:
        return self.status in {"done", "reduced"}

    @property
    def has_validation(self) -> bool:
        return (self.path / "validation" / "summary.json").is_file()


@dataclass(frozen=True)
class ModelInfo:
    trainer: str
    plans: str
    configuration: str
    path: Path
    folds: tuple[FoldInfo, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return f"{self.trainer}__{self.plans}__{self.configuration}"

    @property
    def finished_folds(self) -> tuple[FoldInfo, ...]:
        return tuple(fold for fold in self.folds if fold.finished)

    @property
    def mean_dice(self) -> float | None:
        """Mean validation Dice over the finished cross-validation folds; fold "all" is excluded because it is
        validated on its own training cases."""
        values = [fold.dice for fold in self.folds if fold.name != "all" and fold.finished and fold.dice is not None]
        return sum(values) / len(values) if values else None


def _latest_log(fold_path: Path) -> Path | None:
    logs = list(fold_path.glob("training_log_*.txt"))
    return max(logs, key=lambda path: path.stat().st_mtime_ns) if logs else None


def _log_tail(path: Path, size: int = 64 * 1024) -> str:
    try:
        with path.open("rb") as log:
            log.seek(0, 2)
            log.seek(max(0, log.tell() - size))
            return log.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def summary_foreground_dice(summary: dict | None) -> float | None:
    mean = summary.get("foreground_mean") if isinstance(summary, dict) else None
    value = mean.get("Dice") if isinstance(mean, dict) else None
    return float(value) if isinstance(value, (int, float)) and not math.isnan(value) else None


def fold_info(fold_path: Path, now: float | None = None) -> FoldInfo:
    """Read status, epoch and Dice of one fold from its files."""
    now = time.time() if now is None else now
    name = fold_path.name.removeprefix("fold_")
    final = (fold_path / "checkpoint_final.pth").is_file()
    log = _latest_log(fold_path)
    content = _log_tail(log) if log else ""
    epochs = _EPOCH_LINE.findall(content)
    epoch = int(epochs[-1]) if epochs else None
    dice_lines = _PSEUDO_DICE_LINE.findall(content)
    pseudo = None
    if dice_lines:
        values = [float(value) for value in _NUMBER.findall(dice_lines[-1].replace("np.float32", ""))]
        pseudo = sum(values) / len(values) if values else None
    debug = json_object(fold_path / "debug.json") or {}
    try:
        total = int(debug.get("num_epochs")) if debug.get("num_epochs") is not None else None
    except (TypeError, ValueError):
        total = None
    if final:
        status = "done" if log is not None or (fold_path / "checkpoint_best.pth").is_file() else "reduced"
    elif log is not None and now - log.stat().st_mtime < RUNNING_LOG_AGE_SECONDS:
        status = "training"
    else:
        status = "stopped"
    dice = summary_foreground_dice(json_object(fold_path / "validation" / "summary.json"))
    return FoldInfo(name, fold_path, status, epoch, total, pseudo, dice)


def models_overview(results_path: Path | None) -> tuple[ModelInfo, ...]:
    """Describe every trained model folder of a results dataset with its folds."""
    if results_path is None or not results_path.is_dir():
        return ()
    models = []
    for relative in trained_models(results_path):
        path = results_path / relative
        parts = model_trainer_plan_configuration(path)
        if parts is None:
            continue
        folds = tuple(fold_info(fold) for fold in model_folds(path))
        models.append(ModelInfo(*parts, path, folds))
    return tuple(models)


def any_fold_training(models: tuple[ModelInfo, ...]) -> bool:
    return any(fold.status == "training" for model in models for fold in model.folds)


# ---------------------------------------------------------------- validation

@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    prediction_file: str
    reference_file: str
    mean_dice: float | None
    weakest: tuple[str, float] | None
    metrics: dict


def _mean_dice(metrics: dict) -> float | None:
    values = [entry.get("Dice") for entry in metrics.values() if isinstance(entry, dict)]
    values = [value for value in values if isinstance(value, (int, float)) and not math.isnan(value)]
    return sum(values) / len(values) if values else None


def validation_cases(summary: dict | None, names: dict[int, str] | None = None) -> tuple[ValidationCase, ...]:
    """Turn ``metric_per_case`` into rows with the mean Dice and the weakest class."""
    names = names or {}
    entries = summary.get("metric_per_case") if isinstance(summary, dict) else None
    if not isinstance(entries, list):
        return ()
    rows = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("metrics"), dict):
            continue
        metrics = entry["metrics"]
        reference = str(entry.get("reference_file", ""))
        case = strip_image_suffix(Path(reference).name) if reference else ""
        weakest = None
        for label, values in metrics.items():
            dice = values.get("Dice") if isinstance(values, dict) else None
            if isinstance(dice, (int, float)) and not math.isnan(dice) and (weakest is None or dice < weakest[1]):
                try:
                    label_name = names.get(int(label), str(label))
                except ValueError:
                    label_name = str(label)
                weakest = (label_name, float(dice))
        rows.append(ValidationCase(case, str(entry.get("prediction_file", "")), reference,
                                   _mean_dice(metrics), weakest, metrics))
    return tuple(rows)


def mean_class_dice(summary: dict | None, names: dict[int, str] | None = None) -> list[tuple[str, float]]:
    """Return ``(class name, mean Dice)`` from a summary's ``mean`` section."""
    names = names or {}
    mean = summary.get("mean") if isinstance(summary, dict) else None
    if not isinstance(mean, dict):
        return []
    rows = []
    for label, values in mean.items():
        dice = values.get("Dice") if isinstance(values, dict) else None
        if isinstance(dice, (int, float)) and not math.isnan(dice):
            try:
                rows.append((names.get(int(label), str(label)), float(dice)))
            except ValueError:
                rows.append((str(label), float(dice)))
    return rows


# ---------------------------------------------------------------- geometry and prediction

@dataclass(frozen=True)
class Geometry:
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]


def read_geometry(path: Path) -> Geometry | None:
    """Read size, spacing, origin and direction from the file header only."""
    try:
        import SimpleITK as sitk
    except ImportError:
        return None
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return Geometry(tuple(reader.GetSize()), tuple(reader.GetSpacing()), tuple(reader.GetOrigin()),
                        tuple(reader.GetDirection()))
    except Exception:
        return None


def geometries_match(first: Geometry | None, second: Geometry | None, tolerance: float = 1e-4) -> bool | None:
    if first is None or second is None:
        return None
    if first.shape != second.shape:
        return False
    return all(abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
               for a, b in zip(first.spacing + first.origin + first.direction,
                               second.spacing + second.origin + second.direction))


def prediction_status(input_dir: Path | None, output_dir: Path | None) -> list[tuple[str, bool, bool]]:
    """List input cases as ``(case ID, channels present, prediction exists)``."""
    if input_dir is None or not input_dir.is_dir():
        return []
    cases: dict[str, set[str | None]] = {}
    for path in input_dir.iterdir():
        if path.is_file() and has_image_suffix(path):
            case, channel = _image_case(path)
            cases.setdefault(case, set()).add(channel)
    predicted = set()
    if output_dir is not None and output_dir.is_dir():
        predicted = {strip_image_suffix(path) for path in output_dir.iterdir()
                     if path.is_file() and image_suffix(path)}
    return [(case, "0000" in cases[case] or None in cases[case], case in predicted)
            for case in sorted(cases, key=str.casefold)]
