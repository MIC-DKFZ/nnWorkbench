from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from nnworkbench.services.image_files import has_image_suffix


class PredictionInspectionDialog(QDialog):
    """Choose folders used to compare externally produced predictions."""

    def __init__(self, raw_dataset: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Inspect predictions")
        layout = QFormLayout(self)
        self.image_directory = QLineEdit()
        self.prediction_directory = QLineEdit()
        self.reference_directory = QLineEdit()
        image_source = raw_dataset / "imagesTr" if raw_dataset else None
        reference_source = raw_dataset / "labelsTr" if raw_dataset else None
        layout.addRow("Image folder (optional)", self._directory_field(
            self.image_directory, "Choose image folder", image_source, "imagesTr"))
        layout.addRow("Prediction folder", self._directory_field(self.prediction_directory, "Choose prediction folder"))
        layout.addRow("Reference folder (optional)", self._directory_field(
            self.reference_directory, "Choose reference folder", reference_source, "labelsTr"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Inspect")
        buttons.accepted.connect(self._accept_if_valid); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    @staticmethod
    def _directory_field(field: QLineEdit, title: str, suggested_path: Path | None = None,
                         suggested_text: str | None = None) -> QWidget:
        widget = QWidget(); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: PredictionInspectionDialog._browse_directory(field, title))
        layout.addWidget(field)
        if suggested_text is not None:
            suggested = QPushButton(suggested_text)
            suggested.setEnabled(suggested_path is not None and suggested_path.is_dir())
            if suggested_path is not None:
                suggested.clicked.connect(lambda: field.setText(str(suggested_path)))
            layout.addWidget(suggested)
        layout.addWidget(browse)
        return widget

    @staticmethod
    def _browse_directory(field: QLineEdit, title: str) -> None:
        selected = QFileDialog.getExistingDirectory(field, title, field.text())
        if selected:
            field.setText(selected)

    def directories(self) -> tuple[Path | None, Path, Path | None]:
        optional = lambda text: Path(text).expanduser() if text else None
        return (optional(self.image_directory.text().strip()), Path(self.prediction_directory.text().strip()).expanduser(),
                optional(self.reference_directory.text().strip()))

    def _accept_if_valid(self) -> None:
        image_directory, prediction_directory, reference_directory = self.directories()
        invalid = [directory for directory in (image_directory, prediction_directory, reference_directory)
                   if directory is not None and not directory.is_dir()]
        if invalid:
            QMessageBox.warning(self, "Folder not found", f"Choose existing folders.\n\n{invalid[0]}")
            return
        if not prediction_directory.is_dir():
            QMessageBox.warning(self, "Prediction folder required", "Choose an existing prediction folder.")
            return
        if not any(path.is_file() and has_image_suffix(path) for path in prediction_directory.iterdir()):
            QMessageBox.warning(self, "No predictions found", "The prediction folder must contain prediction images.")
            return
        self.accept()
