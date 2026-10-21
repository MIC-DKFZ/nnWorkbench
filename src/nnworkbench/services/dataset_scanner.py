from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from nnworkbench.models.dataset import DatasetRecord

DATASET_DIRECTORY = re.compile(r"^Dataset(?P<id>\d+)_(?P<name>.+)$")


def discover(root: Path | None) -> dict[int, tuple[str, Path]]:
    """Dataset folders by ID; an unreadable root has none. With two folders of one ID (Dataset0420_A and
    Dataset420_B) the first in name order is used."""
    try:
        if root is None or not root.is_dir():
            return {}
        entries = sorted(root.iterdir(), key=lambda entry: entry.name)
    except OSError:  # no permission, name too long, ...
        return {}
    found: dict[int, tuple[str, Path]] = {}
    for entry in entries:
        match = DATASET_DIRECTORY.match(entry.name)
        if match and int(match["id"]) not in found and _is_dir(entry):
            found[int(match["id"])] = (match["name"], entry)
    return found


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
    return sorted(records, key=lambda record: record.dataset_id)


def id_in_use(dataset_id: int, roots: Iterable[Path | None]) -> bool:
    return any(dataset_id in discover(root) for root in roots)
