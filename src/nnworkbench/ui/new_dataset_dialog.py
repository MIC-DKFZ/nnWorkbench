from __future__ import annotations

from collections.abc import Callable

from qtpy.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit

from nnworkbench.models.dataset import DatasetRecord

from .widgets import NumberField, with_stepper


class NewDatasetDialog(QDialog):
    """A new dataset, or with ``sources`` the next iteration of a chosen source dataset."""

    def __init__(self, parent=None, *, dataset_id: int = 1, name: str = "", title: str = "Create nnU-Net dataset",
                 sources: list[DatasetRecord] | None = None, source: DatasetRecord | None = None,
                 suggest: Callable[[DatasetRecord], tuple[int, str]] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QFormLayout(self)
        self.dataset_id = NumberField(1, 999999, dataset_id)
        self.name = QLineEdit(name); self.name.setPlaceholderText("e.g. LiverCT")
        self._sources = sources or []
        self._suggest = suggest
        self.source_box: QComboBox | None = None
        if self._sources:
            self.source_box = QComboBox()
            for record in self._sources:
                self.source_box.addItem(f"{record.dataset_id:03d} · {record.name}")
            self.source_box.setToolTip("The dataset whose images, labels and dataset.json are copied")
            layout.addRow("Source dataset", self.source_box)
            start = self._sources.index(source) if source in self._sources else 0
            self.source_box.setCurrentIndex(start)
            self.source_box.currentIndexChanged.connect(self._source_changed)
            self._source_changed(start)  # suggested ID and name of the preselected source
        layout.addRow("Dataset ID", with_stepper(self.dataset_id)); layout.addRow("Name", self.name)
        self.description: QCheckBox | None = None
        if not self._sources:  # a new iteration copies the description of its source instead
            self.description = QCheckBox("Add dataset-description.md")
            self.description.setChecked(True)
            self.description.setToolTip("A template to describe your images and define the classes (guide: 1. Describe "
                                        "and 2. Define), with images. Open it on the Data page.")
            layout.addRow("", self.description)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def source(self) -> DatasetRecord | None:
        return self._sources[self.source_box.currentIndex()] if self.source_box is not None else None

    def _source_changed(self, index: int) -> None:
        if self._suggest is not None and 0 <= index < len(self._sources):
            dataset_id, name = self._suggest(self._sources[index])
            self.dataset_id.setValue(dataset_id)
            self.name.setText(name)
