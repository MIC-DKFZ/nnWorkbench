from __future__ import annotations

from qtpy.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit

from .widgets import NumberField, with_stepper


class NewDatasetDialog(QDialog):
    def __init__(self, parent=None, *, dataset_id: int = 1, name: str = "", title: str = "Create nnU-Net dataset") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QFormLayout(self)
        self.dataset_id = NumberField(1, 999999, dataset_id)
        self.name = QLineEdit(name); self.name.setPlaceholderText("e.g. LiverCT")
        layout.addRow("Dataset ID", with_stepper(self.dataset_id)); layout.addRow("Name", self.name)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
