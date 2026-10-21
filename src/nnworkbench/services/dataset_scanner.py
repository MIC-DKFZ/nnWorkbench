from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from nnworkbench.models.dataset import DatasetRecord

DATASET_DIRECTORY = re.compile(r"^Dataset(?P<id>\d+)_(?P<name>.+)$")


def _dataset_folders(root: Path | None) -> list[Path]:
    """The folders in a root whose name starts with "Dataset", in name order; an unreadable root has none."""
    try:
        if root is None or not root.is_dir():
            return []
        entries = sorted(root.iterdir(), key=lambda entry: entry.name)
    except OSError:  # no permission, name too long, ...
        return []
    return [entry for entry in entries if entry.name.startswith("Dataset") and _is_dir(entry)]


def _nnunet_prefix(dataset_id: int) -> str:
    """The folder prefix nnU-Net looks for (find_candidate_datasets: "Dataset%03.0d")."""
    return f"Dataset{dataset_id:03d}"


def discover(root: Path | None) -> dict[int, tuple[str, Path]]:
    """Dataset folders by ID; an unreadable root has none. With two folders of one ID the one nnU-Net uses wins
    (Dataset420_B over Dataset0420_A), otherwise the first in name order; id_problems reports the conflict."""
    found: dict[int, tuple[str, Path]] = {}
    for entry in _dataset_folders(root):
        match = DATASET_DIRECTORY.match(entry.name)
        if not match:
            continue
        dataset_id = int(match["id"])
        if dataset_id not in found or (not found[dataset_id][1].name.startswith(_nnunet_prefix(dataset_id))
                                       and entry.name.startswith(_nnunet_prefix(dataset_id))):
            found[dataset_id] = (match["name"], entry)
    return found


def id_problems(dataset_ids: Iterable[int], roots: Iterable[Path | None]
                ) -> tuple[dict[int, tuple[str, str]], dict[int, tuple[str, ...]]]:
    """As nnU-Net's convert_id_to_dataset_name: it finds a dataset by the folder prefix "Dataset%03d" in all three
    roots and refuses the ID when the folders found have more than one name (Dataset420_A and Dataset420_B, but also
    Dataset4200_X for ID 420). Returns the refused IDs as (badge text, explanation), and per ID the folders that parse
    to it but that nnU-Net never uses (Dataset0420_A: alone it is not found at all, next to Dataset420_B ignored)."""
    names = [entry.name for root in roots for entry in _dataset_folders(root)]
    problems: dict[int, tuple[str, str]] = {}
    ignored: dict[int, tuple[str, ...]] = {}
    for dataset_id in dataset_ids:
        prefix = _nnunet_prefix(dataset_id)
        candidates = sorted({name for name in names if name.startswith(prefix)})  # what nnU-Net finds
        parsed = {name for name in names if (match := DATASET_DIRECTORY.match(name)) and int(match["id"]) == dataset_id}
        if candidates and parsed - set(candidates):
            ignored[dataset_id] = tuple(sorted(parsed - set(candidates)))
        if len(candidates) > 1:
            problems[dataset_id] = (
                "duplicate ID",
                (f"ID {dataset_id} is used by {len(candidates)} folders: {', '.join(candidates)}. nnU-Net refuses "
                 "this ID (\"More than one dataset name found\"); rename or remove all but one, in nnUNet_raw, "
                 "nnUNet_preprocessed and nnUNet_results."))
        elif not candidates and parsed:
            folder = min(parsed)
            name = DATASET_DIRECTORY.match(folder)["name"]
            problems[dataset_id] = (
                "name not found by nnU-Net",
                (f"nnU-Net does not find {folder} by ID {dataset_id}: it looks for folders starting with "
                 f"{prefix}. Rename the folder to {prefix}_{name}."))
    return problems, ignored


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def scan_datasets(raw_root: Path | None, preprocessed_root: Path | None,
                  results_root: Path | None) -> list[DatasetRecord]:
    sources = [("raw_path", discover(raw_root)), ("preprocessed_path", discover(preprocessed_root)),
               ("results_path", discover(results_root))]
    ids = set().union(*(items.keys() for _, items in sources))
    records: list[DatasetRecord] = []
    for dataset_id in ids:
        values: dict[str, Path | None] = {"raw_path": None, "preprocessed_path": None, "results_path": None}
        names: list[str] = []
        for attribute, items in sources:
            if dataset_id in items:
                name, path = items[dataset_id]
                names.append(name)
                values[attribute] = path
        records.append(DatasetRecord(dataset_id, names[0], **values))
    problems, ignored = id_problems(ids, (raw_root, preprocessed_root, results_root))
    records = [replace(record, id_problem=problems.get(record.dataset_id, ("", "")),
                       ignored_folders=ignored.get(record.dataset_id, ())) for record in records]
    return sorted(records, key=lambda record: record.dataset_id)


def id_badge(record: DatasetRecord) -> tuple[str, str, str] | None:
    """(text, tone, explanation) for the dataset's ID badge: red when nnU-Net refuses the ID, orange when nnU-Net
    works but ignores a folder of this ID, None when all is well."""
    if record.id_problem[0]:
        return record.id_problem[0], "red", record.id_problem[1]
    if record.ignored_folders:
        used = (record.raw_path or record.preprocessed_path or record.results_path)
        return ("folder ignored by nnU-Net", "orange",
                (f"nnU-Net uses {used.name if used else record.directory_name} for ID {record.dataset_id} and "
                 f"ignores {', '.join(record.ignored_folders)}: it looks for folders starting with "
                 f"{_nnunet_prefix(record.dataset_id)}. Rename or remove the ignored folder."))
    return None


def id_in_use(dataset_id: int, roots: Iterable[Path | None]) -> bool:
    return any(dataset_id in discover(root) for root in roots)
