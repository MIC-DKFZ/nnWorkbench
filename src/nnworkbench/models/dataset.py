from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: int
    name: str
    raw_path: Path | None = None
    preprocessed_path: Path | None = None
    results_path: Path | None = None
    # A problem with the dataset ID that nnU-Net refuses (see dataset_scanner.id_problems): short badge text and
    # the explanation; empty when there is none.
    id_problem: tuple[str, str] = ("", "")
    # Folders that parse to this ID but that nnU-Net never uses (Dataset0420_A next to Dataset420_B).
    ignored_folders: tuple[str, ...] = ()

    @property
    def directory_name(self) -> str:
        return f"Dataset{self.dataset_id:03d}_{self.name}"
