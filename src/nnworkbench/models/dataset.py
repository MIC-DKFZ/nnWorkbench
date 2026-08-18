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

    @property
    def directory_name(self) -> str:
        return f"Dataset{self.dataset_id:03d}_{self.name}"

    @property
    def has_raw_metadata(self) -> bool:
        return self.raw_path is not None and (self.raw_path / "dataset.json").is_file()
