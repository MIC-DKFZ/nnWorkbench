"""Read-only status of a dataset for the stage pages: cases, plans, models and validation.

Everything here only lists folders and reads small JSON or log files. Image data is never
read, so the overview stays fast even for large datasets on network drives.
"""

from __future__ import annotations

import math
import os
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
from .dataset_json import listed_cases
from .image_files import color_components, has_image_suffix, image_suffix, strip_image_suffix

_CHANNEL_SUFFIX = re.compile(r"_(\d{4})$")
_EPOCH_LINE = re.compile(r"(?m)^.*?:\s*Epoch\s+(\d+)\s*$")
UNSUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg")  # common, but no nnU-Net reader
_EPOCH_TIME_LINE = re.compile(r"(?m)^.*?:\s*Epoch time:\s*([0-9.]+)\s*s\s*$")
_PSEUDO_DICE_LINE = re.compile(r"(?m)^.*?:\s*Pseudo dice\s*\[(.*)\]\s*$")
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# A fold counts as "training" while its log changed recently; afterwards it is "stopped".
RUNNING_LOG_AGE_SECONDS = 15 * 60


def finite_number(value: object) -> float | None:
    """A JSON or log number as a float; ``None`` for anything else, NaN, infinity and integers too large for a
    float (e.g. ``1e999`` or a 400-digit Dice in a hand-edited summary.json)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _scan(folder: Path | None) -> list[os.DirEntry]:
    """The entries of a folder with their type from the listing itself (os.scandir: no stat call per entry on most
    file systems, which matters on network drives); empty when the folder is missing or cannot be read."""
    if folder is None:
        return []
    try:
        with os.scandir(folder) as entries:
            return list(entries)
    except (OSError, ValueError):
        return []


def _entry_is_file(entry: os.DirEntry) -> bool:
    try:
        return entry.is_file()
    except OSError:
        return False


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:  # e.g. a fold folder without read permission (is_file raises PermissionError)
        return False


def _modified(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except (OSError, ValueError):  # broken symlink, removed meanwhile
        return None


def _numbers(value: object) -> tuple:
    """A list from a plans file (patch size, spacing) as a tuple; anything else is not shown."""
    return tuple(value) if isinstance(value, list) else ()


# ---------------------------------------------------------------- dataset.json helpers

def ignore_label_id(manifest: dict | None) -> int | None:
    """Return the ``ignore`` label ID, or ``None`` for dense labels.

    As in nnU-Net, only the exact key ``"ignore"`` counts: "Ignore" or " ignore " is a normal class there
    (``definition_errors`` reports it as a dataset.json problem)."""
    labels = manifest.get("labels") if isinstance(manifest, dict) else None
    if not isinstance(labels, dict):
        return None
    value = labels.get("ignore")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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
    channels_present: bool  # a file for every channel of dataset.json (RGB files count as three)
    label_file: Path | None
    last_edit: float | None  # modification time of the label file
    channel_suffixes: tuple[str | None, ...]  # of image_files ("0000", …; None without a suffix), read once

    @property
    def has_label(self) -> bool:
        return self.label_file is not None

    def extra_channels(self, required: tuple[str, ...]) -> list[str]:
        """Channel suffixes of this case beyond channel_names (nnU-Net: "Unexpected number of modalities")."""
        return [channel for channel in self.channel_suffixes if channel is not None and channel not in required]


@dataclass(frozen=True)
class RawOverview:
    cases: tuple[CaseRow, ...]
    image_files: int
    label_files: int
    channels: int
    names_valid: bool
    unmatched_labels: tuple[str, ...] = ()  # label files without an image of that case
    unsupported_images: tuple[str, ...] = ()  # e.g. .jpg in imagesTr, which nnU-Net cannot read
    # the channel suffixes a case needs (_0000 … ; one RGB file covers three channel names)
    required_suffixes: tuple[str, ...] = ("0000",)
    channel_problem: str = ""  # colour channels of the files do not add up to channel_names
    file_list: bool = False  # the cases come from the "dataset" file list of dataset.json, not imagesTr/labelsTr
    missing_files: tuple[str, ...] = ()  # images listed there but not found (a missing label: missing_labels)
    unlisted_images: tuple[str, ...] = ()  # in imagesTr but not in the file list: nnU-Net ignores them
    label_times: tuple[float, ...] = ()  # when each label file found was last changed (the outdated checks)

    @property
    def newest_label(self) -> float | None:
        """The last label change; a time from the future (clock skew) is ignored, it would make preprocessing
        outdated forever, even right after it."""
        times = [value for value in self.label_times if value <= time.time() + 60]
        return max(times) if times else None

    @property
    def missing_labels(self) -> int:
        return sum(not case.has_label for case in self.cases)

    @property
    def channels_complete(self) -> bool:
        return all(case.channels_present for case in self.cases)


def _image_case(path: str | Path) -> tuple[str, str | None]:
    """Split an imagesTr file name into case ID and channel (``"0000"``), if present."""
    stem = strip_image_suffix(path)
    match = _CHANNEL_SUFFIX.search(stem)
    return (stem[:match.start()], match.group(1)) if match else (stem, None)


def required_channels(channels: int, components: dict[str, int]) -> tuple[tuple[str, ...], str]:
    """The channel suffixes each case needs for ``channels`` channel names, when the files of a suffix hold
    ``components`` channels each (RGB: 3; read from the image header), and what does not add up ("" if all does).
    As nnU-Net's 2D reader: a case's channels are the colour channels of its files, in suffix order."""
    required, total, index = [], 0, 0
    while total < channels:
        suffix = f"{index:04d}"
        required.append(suffix)
        total += components.get(suffix, 1)
        index += 1
    odd = {suffix: count for suffix, count in components.items() if count not in (1, 3, 4)}
    if odd:
        suffix, count = next(iter(odd.items()))
        return tuple(required), (f"_{suffix} images have {count} colour channels (grey + alpha?); nnU-Net reads 1 "
                                 "(grey), 3 (RGB) or 4 (RGBA): save them as greyscale or RGB")
    if total > channels:
        colour = {suffix: count for suffix, count in components.items() if count > 1 and suffix in required}
        shown = ", ".join(f"_{suffix} has {count} (RGB{'A' if count == 4 else ''})" for suffix, count in colour.items())
        return tuple(required), (f"the images have {total} colour channels ({shown}), but dataset.json lists "
                                 f"{channels} in channel_names: name each colour channel (e.g. R, G, B) or save the "
                                 "images as greyscale")
    return tuple(required), ""


def _listed_overview(raw_path: Path, manifest: dict, listed: dict[str, tuple[tuple[Path, ...], Path]]) -> RawOverview:
    """The cases of dataset.json's "dataset" file list (nnU-Net then ignores imagesTr/labelsTr). A case's images
    are its channels in list order (an RGB file counts three, from the header of the first case's files)."""
    channels = channel_count(manifest)
    first = next(iter(listed.values()), ((), None))[0]
    components = {f"{index:04d}": color_components(path) or 1 for index, path in enumerate(first) if _is_file(path)}
    required, channel_problem = required_channels(channels, components)
    rows, missing, label_times = [], [], []
    image_count = label_count = 0
    for case in sorted(listed, key=str.casefold):
        images, label = listed[case]
        present = tuple(path for path in images if _is_file(path))
        missing += [str(path) for path in images if path not in present]
        image_count += len(present)
        label_file = label if _is_file(label) else None  # none yet: "without label", e.g. still to annotate
        label_count += label_file is not None
        edited = _modified(label_file) if label_file else None
        if edited is not None:
            label_times.append(edited)
        rows.append(CaseRow(case, present, len(present) == len(images) == len(required), label_file, edited,
                            tuple(_image_case(path.name)[1] for path in present)))
    listed_files = {path for images, _label in listed.values() for path in images}
    unlisted = sorted((entry.name for entry in _scan(raw_path / "imagesTr")
                       if _entry_is_file(entry) and has_image_suffix(entry.name)
                       and Path(os.path.abspath(entry.path)) not in listed_files), key=str.casefold)
    return RawOverview(tuple(rows), image_count, label_count, channels, True, (), (), required, channel_problem,
                       True, tuple(missing), tuple(unlisted), tuple(label_times))


def raw_overview(raw_path: Path | None, manifest: dict | None) -> RawOverview:
    """List imagesTr and labelsTr and match them into cases (or the "dataset" file list of dataset.json)."""
    listed = listed_cases(raw_path, manifest) if raw_path is not None and isinstance(manifest, dict) else None
    if listed is not None:
        return _listed_overview(raw_path, manifest, listed)
    channels = channel_count(manifest)
    images_dir = raw_path / "imagesTr" if raw_path else None
    labels_dir = raw_path / "labelsTr" if raw_path else None
    images: dict[str, list[Path]] = {}
    names_valid = True
    image_count = 0
    unsupported = []
    for entry in _scan(images_dir):
        if not _entry_is_file(entry):
            continue
        if not has_image_suffix(entry.name):
            if os.path.splitext(entry.name)[1].lower() in UNSUPPORTED_IMAGE_SUFFIXES:
                unsupported.append(entry.name)
            continue
        image_count += 1
        case, channel = _image_case(entry.name)
        if channel is None:
            names_valid = False
        images.setdefault(case, []).append(Path(entry.path))
    labels: dict[str, Path] = {}
    for entry in _scan(labels_dir):
        if _entry_is_file(entry) and has_image_suffix(entry.name):
            labels[strip_image_suffix(entry.name)] = Path(entry.path)
    # one stat per label file, shared by the case rows and the outdated checks (newest_label, label_times)
    edited = {case: _modified(path) for case, path in labels.items()}
    # colour channels per suffix, from the first image of each (2D formats only; one header read per suffix)
    components: dict[str, int] = {}
    for case in sorted(images, key=str.casefold):
        for path in images[case]:
            suffix = _image_case(path)[1]
            if suffix is not None and suffix not in components:
                components[suffix] = color_components(path) or 1
    required, channel_problem = required_channels(channels, components)
    rows = []
    for case in sorted(images, key=str.casefold):
        files = tuple(sorted(images[case], key=lambda item: item.name))
        suffixes = tuple(_image_case(item.name)[1] for item in files)
        rows.append(CaseRow(case, files, set(required) <= set(suffixes), labels.get(case), edited.get(case),
                            suffixes))
    unmatched = tuple(sorted((path.name for case, path in labels.items() if case not in images), key=str.casefold))
    return RawOverview(tuple(rows), image_count, len(labels), channels, names_valid, unmatched,
                       tuple(sorted(unsupported, key=str.casefold)), required, channel_problem,
                       label_times=tuple(value for value in edited.values() if value is not None))


def newest_label_time(raw_path: Path | None) -> float | None:
    """The newest change of a label file (in labelsTr, or listed in dataset.json), for the outdated rules: the
    case listing's ``RawOverview.newest_label``, which ``load_state`` already has."""
    if raw_path is None:
        return None
    return raw_overview(raw_path, json_object(raw_path / "dataset.json")).newest_label


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
    # .b2nd since nnU-Net 2.6; .npz (with .npy) from older versions, which nnU-Net still trains on
    return sum(1 for entry in _scan(folder) if _entry_is_file(entry) and (
        entry.name.endswith(".b2nd") and not entry.name.endswith("_seg.b2nd") or entry.name.endswith(".npz")))


def _inherited(configurations: dict, name: str) -> dict:
    """A configuration with the values of its ``inherits_from`` parents (3d_cascade_fullres uses 3d_fullres's data),
    as nnU-Net's plans handler resolves it."""
    values: dict = {}
    seen = set()
    while isinstance(name, str) and isinstance(configurations.get(name), dict) and name not in seen:
        seen.add(name)
        values = {**configurations[name], **values}
        name = configurations[name].get("inherits_from")
    return values


def plans_overview(preprocessed_path: Path | None, raw_path: Path | None = None,
                   raw: RawOverview | None = None) -> tuple[PlansInfo, ...]:
    """Describe every plans file of a preprocessed dataset and its configurations. ``raw`` (the case listing) has
    the newest label change already; without it the labels are listed here."""
    if preprocessed_path is None or not preprocessed_path.is_dir():
        return ()
    newest_label = raw.newest_label if raw is not None else newest_label_time(raw_path)
    plans: list[PlansInfo] = []
    for file in sorted(preprocessed_path.glob("*.json"), key=lambda item: item.name.casefold()):
        content = json_object(file)
        if not content or not isinstance(content.get("configurations"), dict):
            continue
        configurations = []
        times = []
        for name, values in content["configurations"].items():
            values = _inherited(content["configurations"], name)
            if not values:
                continue
            identifier = values.get("data_identifier") or f"{file.stem}_{name}"
            folder = preprocessed_path / str(identifier)
            count = _count_preprocessed_cases(folder)
            if count and (modified := _modified(folder)) is not None:
                times.append(modified)
            batch_size = values.get("batch_size")
            configurations.append(ConfigurationInfo(
                str(name), _numbers(values.get("patch_size")), _numbers(values.get("spacing")),
                batch_size if isinstance(batch_size, int) and not isinstance(batch_size, bool) else None,
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
    log_time: float | None = None  # last change of the training log

    @property
    def finished(self) -> bool:
        return self.status in {"done", "reduced"}

    def log_note(self, now: float | None = None) -> str:
        """How sure "training" or "stopped" is: both are read from the fold folder (the training may run on another
        computer), from how long ago the log changed. "" for finished folds or without a log."""
        if self.finished or self.log_time is None:
            return ""
        age = (time.time() if now is None else now) - self.log_time
        if self.status == "training":
            return f"log updated {format_age(age)} ago: probably still training"
        return (f"no log update for {format_age(age)}: stopped, unless it runs somewhere that has not written the "
                "log yet" if age >= 0 else "log time in the future (clock difference?): status unknown")

    @property
    def has_validation(self) -> bool:
        return _is_file(self.path / "validation" / "summary.json")


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
        return finite_number(sum(values) / len(values)) if values else None


def _latest_log(fold_path: Path) -> tuple[Path, float] | None:
    """The newest training log with its modification time (broken symlinks and unreadable logs are skipped)."""
    try:
        logs = list(fold_path.glob("training_log_*.txt"))
    except OSError:
        return None
    dated = [(modified, path) for path in logs if (modified := _modified(path)) is not None]
    if not dated:
        return None
    modified, path = max(dated, key=lambda item: item[0])
    return path, modified


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
    return finite_number(mean.get("Dice") if isinstance(mean, dict) else None)


def _integer(text: object) -> int | None:
    """An epoch count from a log or debug.json; ``None`` when it is not a number of a sensible size."""
    if not isinstance(text, str) or not text.strip().isdecimal() or len(text.strip()) > 18:
        return None
    return int(text)


def format_age(seconds: float) -> str:
    """A rough duration: "40 s", "3 min", "2 h", "5 days"."""
    seconds = abs(seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.0f} h"
    return f"{seconds / 86400:.0f} days"


def _running_window(log: str) -> float:
    """How long a training may stay silent: 15 minutes, or two epochs when epochs are slower (CPU, Apple GPU)."""
    times = _EPOCH_TIME_LINE.findall(log)
    try:
        epoch = float(times[-1]) if times else 0.0
    except ValueError:
        epoch = 0.0
    return max(RUNNING_LOG_AGE_SECONDS, 2 * epoch if epoch < 7 * 24 * 3600 else 0)


def fold_info(fold_path: Path, now: float | None = None, ended: float | None = None) -> FoldInfo:
    """Read status, epoch and Dice of one fold from its files. ``ended``: when a job of this GUI that trained the
    fold ended; a log not written after that is no training (e.g. it failed at start)."""
    now = time.time() if now is None else now
    name = fold_path.name.removeprefix("fold_")
    final = _is_file(fold_path / "checkpoint_final.pth")
    latest = _latest_log(fold_path)
    log, log_time = latest if latest else (None, None)
    content = _log_tail(log) if log else ""
    epochs = _EPOCH_LINE.findall(content)
    epoch = _integer(epochs[-1]) if epochs else None
    dice_lines = _PSEUDO_DICE_LINE.findall(content)
    pseudo = None
    if dice_lines:
        values = [finite_number(float(value)) for value in _NUMBER.findall(dice_lines[-1].replace("np.float32", ""))]
        values = [value for value in values if value is not None]
        pseudo = finite_number(sum(values) / len(values)) if values else None
    debug = json_object(fold_path / "debug.json") or {}
    epochs_value = debug.get("num_epochs")
    total = int(epochs_value) if finite_number(epochs_value) is not None else _integer(epochs_value)
    if final:
        status = "done" if log is not None or _is_file(fold_path / "checkpoint_best.pth") else "reduced"
    elif log_time is not None and abs(now - log_time) < _running_window(content) and not (
            ended is not None and log_time <= ended + 5):
        status = "training"  # a log far in the future (clock skew, copied files) is not a running training
    else:
        status = "stopped"
    dice = summary_foreground_dice(json_object(fold_path / "validation" / "summary.json"))
    return FoldInfo(name, fold_path, status, epoch, total, pseudo, dice, log_time)


def models_overview(results_path: Path | None, ended: dict[Path, float] | None = None) -> tuple[ModelInfo, ...]:
    """Describe every trained model folder of a results dataset with its folds (``ended``: fold folder → when a
    GUI job training it ended, see fold_info)."""
    if results_path is None or not results_path.is_dir():
        return ()
    models = []
    for relative in trained_models(results_path):
        path = results_path / relative
        parts = model_trainer_plan_configuration(path)
        if parts is None:
            continue
        folds = tuple(fold_info(fold, ended=(ended or {}).get(fold)) for fold in model_folds(path))
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
    values = [finite_number(entry.get("Dice")) for entry in metrics.values() if isinstance(entry, dict)]
    values = [value for value in values if value is not None]
    return finite_number(sum(values) / len(values)) if values else None


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
            dice = finite_number(values.get("Dice") if isinstance(values, dict) else None)
            if dice is not None and (weakest is None or dice < weakest[1]):
                weakest = (class_name(label, names), dice)
        rows.append(ValidationCase(case, str(entry.get("prediction_file", "")), reference,
                                   _mean_dice(metrics), weakest, metrics))
    return tuple(rows)


def class_name(label: object, names: dict[int, str]) -> str:
    """The class name for a label key of summary.json ("1"); regions ("(1, 2)") and unknown keys stay as they are."""
    text = str(label)
    return names.get(int(text), text) if text.isascii() and text.isdigit() else text


def mean_class_dice(summary: dict | None, names: dict[int, str] | None = None) -> list[tuple[str, float]]:
    """Return ``(class name, mean Dice)`` from a summary's ``mean`` section."""
    names = names or {}
    mean = summary.get("mean") if isinstance(summary, dict) else None
    if not isinstance(mean, dict):
        return []
    rows = []
    for label, values in mean.items():
        dice = finite_number(values.get("Dice") if isinstance(values, dict) else None)
        if dice is not None:
            rows.append((class_name(label, names), dice))
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
    except Exception:  # noqa: BLE001  (unreadable image header: no geometry)
        return None


def geometries_match(first: Geometry | None, second: Geometry | None, tolerance: float = 1e-4) -> bool | None:
    if first is None or second is None:
        return None
    if first.shape != second.shape:
        return False
    return all(abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
               for a, b in zip(first.spacing + first.origin + first.direction,
                               second.spacing + second.origin + second.direction))


class PredictionCase(tuple):
    """An input case for prediction: ``(case ID, ready, predicted)`` (equal to that plain tuple), plus the channel
    suffixes found (``"0000"``, …; ``None`` for a file without one), the channels of dataset.json that have no
    file, whether a file has no channel suffix, and the files whose ending is not file_ending. nnU-Net needs one
    file per channel, each with its suffix, and lists only files ending exactly with file_ending."""

    found: frozenset
    missing: tuple[str, ...]
    unsuffixed: bool
    wrong_ending: tuple[str, ...]

    def __new__(cls, case_id: str, found, predicted: bool, channels: int | tuple[str, ...] = 1, wrong_ending=()):
        """``channels``: the number of channels (one file each) or the suffixes needed (RawOverview.required_suffixes,
        where an RGB file covers three channels)."""
        found = frozenset(found)
        needed = (channels if isinstance(channels, tuple)
                  else tuple(f"{index:04d}" for index in range(max(1, channels))))
        missing = tuple(suffix for suffix in needed if suffix not in found)
        unsuffixed = None in found
        wrong_ending = tuple(wrong_ending)
        case = super().__new__(cls, (case_id, not missing and not unsuffixed and not wrong_ending, predicted))
        case.found, case.missing, case.unsuffixed, case.wrong_ending = found, missing, unsuffixed, wrong_ending
        return case

    @property
    def case_id(self) -> str:
        return self[0]

    @property
    def ready(self) -> bool:
        return self[1]

    @property
    def predicted(self) -> bool:
        return self[2]

    def for_channels(self, channels: int | tuple[str, ...]) -> PredictionCase:
        """The same case checked against the channel count of dataset.json."""
        return PredictionCase(self[0], self.found, self[2], channels, self.wrong_ending)


def prediction_status(input_dir: Path | None, output_dir: Path | None, channels: int = 1,
                      file_ending: str | None = None) -> list[PredictionCase]:
    """List input cases as ``(case ID, ready, prediction exists)``: ready when every channel of dataset.json
    (``channels``) has its file (``case_0000``, ``case_0001``, …), no file of the case lacks the suffix and every
    file ends with ``file_ending`` (nnU-Net skips others silently, e.g. ``.TIF`` in a ``.tif`` dataset)."""
    if input_dir is None or not input_dir.is_dir():
        return []
    cases: dict[str, set[str | None]] = {}
    wrong: dict[str, list[str]] = {}
    with os.scandir(input_dir) as entries:  # an unreadable input folder raises: the stage shows the error
        for entry in entries:
            if _entry_is_file(entry) and has_image_suffix(entry.name):
                case, channel = _image_case(entry.name)
                cases.setdefault(case, set()).add(channel)
                if file_ending and not entry.name.endswith(file_ending):
                    wrong.setdefault(case, []).append(entry.name)
    predicted = {strip_image_suffix(entry.name) for entry in _scan(output_dir)
                 if _entry_is_file(entry) and image_suffix(entry.name)}
    return [PredictionCase(case, cases[case], case in predicted, channels, wrong.get(case, ()))
            for case in sorted(cases, key=str.casefold)]
