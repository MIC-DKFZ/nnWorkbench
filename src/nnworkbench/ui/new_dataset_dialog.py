from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QSpinBox


class NewDatasetDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create nnU-Net dataset")
        layout = QFormLayout(self)
        self.dataset_id = QSpinBox(); self.dataset_id.setRange(1, 999999)
        self.name = QLineEdit(); self.name.setPlaceholderText("e.g. LiverCT")
        layout.addRow("Dataset ID", self.dataset_id); layout.addRow("Name", self.name)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
