from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def manifest_source(raw_directory: Path | None, preprocessed_directory: Path | None,
                    results_directory: Path | None) -> tuple[str, Path] | None:
    """Return the first available manifest source in display-priority order."""
    for kind, directory in (("raw", raw_directory), ("preprocessed", preprocessed_directory),
                            ("results", results_directory)):
        if directory and (directory / "dataset.json").is_file():
            return kind, directory
    return None


def num_training_error(manifest: dict, cases: int) -> str | None:
    """nnU-Net's integrity check requires numTraining to equal the number of training cases in imagesTr."""
    value = manifest.get("numTraining")
    if isinstance(value, bool) or not isinstance(value, int):
        return f"numTraining is missing or not a number; it must be the number of training cases ({cases})."
    if value != cases:
        return (f"numTraining is {value}, but imagesTr has {cases} training case{'s' if cases != 1 else ''}. "
                "nnU-Net's dataset check will fail.")
    return None


REGIONS_NOTE = ("Region-based labels (a label is a list of IDs, with regions_class_order): the nnU-Net Workflow "
                "works; annotating and editing the classes in nnWorkbench is not available.")


def has_regions(manifest: object) -> bool:
    """nnU-Net's region-based training: regions_class_order, or a label that is a list of IDs."""
    if not isinstance(manifest, dict):
        return False
    labels = manifest.get("labels")
    return "regions_class_order" in manifest or (
        isinstance(labels, dict) and any(isinstance(value, list) for value in labels.values()))


def definition_errors(manifest: dict) -> tuple[str, ...]:
    """Return user-facing errors for incomplete nnU-Net channel or label definitions."""
    errors: list[str] = []
    channels = manifest.get("channel_names")
    if not isinstance(channels, dict) or not channels:
        errors.append("Channels are not defined. Add continuous channel IDs and names, starting at 0.")
    else:
        channel_ids = list(channels)
        # ASCII digits only: "²" passes isdigit() but not int(), "٠" passes int() but nnU-Net expects "0"
        if (not all(isinstance(identifier, str) and identifier.isascii() and identifier.isdigit()
                    for identifier in channel_ids)
                or not all(isinstance(name, str) and name.strip() for name in channels.values())
                or sorted(int(identifier) for identifier in channel_ids) != list(range(len(channel_ids)))):
            errors.append("Channels must have continuous numeric IDs and a name for every channel.")
    labels = manifest.get("labels")
    if not isinstance(labels, dict) or not labels:
        errors.append("Labels are not defined. Add continuous label IDs and names, starting at 0.")
    elif "regions_class_order" in manifest:
        # region-based training (e.g. BraTS): a label is an ID or a list of IDs, and regions may overlap
        def ids(value) -> list:
            return value if isinstance(value, list) else [value]

        if not all(isinstance(name, str) and name.strip() for name in labels) or not all(
                ids(value) and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                                   for item in ids(value)) for value in labels.values()):
            errors.append("Labels must have a name and a label ID (or a list of IDs for regions).")
    else:
        label_ids = list(labels.values())
        if (not all(isinstance(name, str) and name.strip() for name in labels)
                or not all(isinstance(identifier, int) and not isinstance(identifier, bool) and identifier >= 0
                           for identifier in label_ids)
                or sorted(label_ids) != list(range(len(label_ids)))):
            errors.append("Labels must have continuous numeric IDs and a name for every label.")
    if isinstance(labels, dict):
        # nnU-Net recognises the ignore label only by the exact key "ignore"; anything else is trained as a class
        for name in labels:
            if isinstance(name, str) and name != "ignore" and name.strip().casefold() == "ignore":
                errors.append(f'Label "{name}" is not an ignore label for nnU-Net, which only accepts the exact key '
                              '"ignore" (lower case, no spaces) and would train it as a normal class. Rename it to '
                              '"ignore" for sparse labels, or give the class another name.')
    return tuple(errors)


def file_ending_error(manifest: dict) -> str | None:
    ending = manifest.get("file_ending")
    if not isinstance(ending, str) or not ending.startswith(".") or len(ending) < 2:
        return 'file_ending is missing or invalid; it must be the image file ending, e.g. ".nii.gz".'
    return None


def read_manifest(dataset_directory: Path) -> dict:
    with (dataset_directory / "dataset.json").open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("dataset.json must contain a JSON object")  # noqa: TRY004  (invalid dataset.json content; callers catch ValueError)
    return data


def validate_manifest(data: dict) -> None:
    if not isinstance(data.get("channel_names", {}), dict):
        raise ValueError("channel_names must be an object")  # noqa: TRY004  (invalid dataset.json content; callers catch ValueError)
    if not isinstance(data.get("labels", {}), dict):
        raise ValueError("labels must be an object")  # noqa: TRY004  (invalid dataset.json content; callers catch ValueError)
    if not isinstance(data.get("file_ending", ""), str):
        raise ValueError("file_ending must be text")  # noqa: TRY004  (invalid dataset.json content; callers catch ValueError)
    for field in ("description", "licence"):
        if field in data and not isinstance(data[field], str):
            raise ValueError(f"{field} must be text")


def write_manifest(dataset_directory: Path, data: dict) -> None:
    """Atomically write a raw dataset manifest without creating a backup copy."""
    validate_manifest(data)
    target = dataset_directory / "dataset.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".dataset.", suffix=".json", dir=dataset_directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
