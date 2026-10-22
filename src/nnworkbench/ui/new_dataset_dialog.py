from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.dataset_creator import MAX_NAME_LENGTH, directory_name

from .widgets import ElidedLabel, InfoIcon, NumberField, with_stepper

_MAX_WIDTH = 600


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
        self.name.setMaxLength(MAX_NAME_LENGTH)
        self.name.setCursorPosition(0)
        self.setMinimumWidth(460)
        self.setMaximumWidth(_MAX_WIDTH)  # long names: the folder line ends with "…", the dialog stays on screen
        self.folder = _FolderLabel()  # the folder name: only A-Z and 0-9 are kept
        self.folder.setProperty("muted", True)
        font = self.folder.font()
        font.setPixelSize(12)
        self.folder.setFont(font)
        self.folder.setTextFormat(Qt.TextFormat.PlainText)
        self._sources = sources or []
        self._suggest = suggest
        self.source_box: QComboBox | None = None
        if self._sources:
            self.source_box = QComboBox()
            # as wide as the dialog, not as the longest name (the list shows names with "…", full on hover)
            self.source_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.source_box.setMinimumContentsLength(24)
            self.source_box.view().setTextElideMode(Qt.TextElideMode.ElideRight)
            for record in self._sources:
                self.source_box.addItem(f"{record.dataset_id:03d} · {record.name}")
                self.source_box.setItemData(self.source_box.count() - 1, record.name, Qt.ItemDataRole.ToolTipRole)
            self.source_box.setToolTip("The dataset whose images, labels and dataset.json are copied (or the images "
                                       "linked); the new dataset.json names it in \"inherits_from\"")
            layout.addRow("Source dataset", self.source_box)
            start = self._sources.index(source) if source in self._sources else 0
            self.source_box.setCurrentIndex(start)
            self.source_box.currentIndexChanged.connect(self._source_changed)
            self._source_changed(start)  # suggested ID and name of the preselected source
        layout.addRow("Dataset ID", with_stepper(self.dataset_id))
        # the resulting folder name right under the name (only A-Z and 0-9 are kept), no row of its own
        name_box = QWidget()
        name_column = QVBoxLayout(name_box)
        name_column.setContentsMargins(0, 0, 0, 0)
        name_column.setSpacing(3)
        name_column.addWidget(self.name)
        name_column.addWidget(self.folder)
        layout.addRow("Name", name_box)
        self.dataset_id.valueChanged.connect(self._update_folder)
        self.name.textChanged.connect(self._update_folder)
        self.description: QCheckBox | None = None
        self.link_images: QCheckBox | None = None
        if self._sources:  # where the images of the new round are
            self.link_images = QCheckBox()
            title = QWidget()  # "Link images (i)" as the row's label
            title_row = QHBoxLayout(title)
            title_row.setContentsMargins(0, 0, 0, 0)
            title_row.setSpacing(5)
            title_row.addWidget(QLabel("Link images"))
            title_row.addWidget(InfoIcon("For a round where only the labels change: the images are not copied, the "
                                         "new dataset.json points to them in the source dataset. Labels are always "
                                         "copied."))
            layout.addRow(title, self.link_images)
        if not self._sources:  # a new iteration copies the description of its source instead
            self.description = QCheckBox("Add dataset-description.md")
            self.description.setChecked(True)
            self.description.setToolTip("A template to describe your images and define the classes (guide: 1. Describe "
                                        "and 2. Define), with images. Open it on the Data page.")
            layout.addRow("", self.description)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._update_folder()
        self.name.setFocus()  # the ID is prefilled with a free one; the name is what users type

    def source(self) -> DatasetRecord | None:
        return self._sources[self.source_box.currentIndex()] if self.source_box is not None else None

    def _source_changed(self, index: int) -> None:
        if self._suggest is not None and 0 <= index < len(self._sources):
            dataset_id, name = self._suggest(self._sources[index])
            self.dataset_id.setValue(dataset_id)
            self.name.setText(name)
            self.name.setCursorPosition(0)  # a long suggested name shows its start

    def _update_folder(self, *_args) -> None:
        valid = True
        try:
            text = directory_name(self.dataset_id.value(), self.name.text())
        except ValueError as error:
            valid = False
            text = str(error)
            if self.name.text().strip() and "letters or numbers" in text:  # e.g. a name in Chinese only
                text = "Only A–Z and 0–9 are kept: add Latin letters or numbers"
        # a long folder name keeps its start and end, a message its start
        self.folder.set_mode(Qt.TextElideMode.ElideMiddle if valid else Qt.TextElideMode.ElideRight)
        self.folder.setText(text)
        if hasattr(self, "_ok"):  # not yet while the dialog is built
            self._ok.setEnabled(valid)  # the folder could not be created


class _FolderLabel(ElidedLabel):
    """The folder name, shortened in the middle when it is long; ``text()`` is the full name."""

    def __init__(self) -> None:
        super().__init__(mode=Qt.TextElideMode.ElideMiddle)

    def text(self) -> str:
        return self._full

    def set_mode(self, mode) -> None:
        self._mode = mode
