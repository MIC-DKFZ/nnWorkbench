from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QToolTip,
    QWidget,
)

from nnworkbench.services.commands import prediction_command

from .editable_dropdown import EditableDropdownComboBox


class PredictionDialog(QDialog):
    """Collect parameters required to run prediction from an available trained model."""

    def __init__(self, dataset_id: int, plans_and_configurations: list[tuple[str, str]], parent=None) -> None:
        super().__init__(parent)
        self.dataset_id = dataset_id
        self.setWindowTitle("Run prediction")
        self._options = sorted(set(plans_and_configurations))
        layout = QFormLayout(self)
        self.input_directory = QLineEdit(); self.output_directory = QLineEdit()
        self.plan = EditableDropdownComboBox(); self.plan.addItems(sorted({plan for plan, _ in self._options}))
        self.configuration = EditableDropdownComboBox()
        self.additional_arguments = QLineEdit()
        self.additional_arguments.setPlaceholderText("Optional, e.g. --save_probabilities")
        self.additional_arguments.setToolTip("Extra arguments passed directly to nnUNetv2_predict.")
        self.plan.currentTextChanged.connect(self._set_configurations)
        self._set_configurations(self.plan.currentText())
        layout.addRow("Input directory", self._directory_field(self.input_directory, "Choose input directory"))
        layout.addRow("Output directory", self._directory_field(self.output_directory, "Choose output directory"))
        layout.addRow("Plans", self.plan)
        layout.addRow("Configuration", self.configuration)
        layout.addRow("Additional arguments", self.additional_arguments)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Run prediction")
        self.copy = buttons.addButton("Copy command", QDialogButtonBox.ButtonRole.ActionRole)
        self.copy.clicked.connect(self._copy_command)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _directory_field(self, field: QLineEdit, title: str) -> QWidget:
        widget = QWidget(); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_directory(field, title))
        layout.addWidget(field); layout.addWidget(browse)
        return widget

    def _browse_directory(self, field: QLineEdit, title: str) -> None:
        selected = QFileDialog.getExistingDirectory(self, title, field.text())
        if selected:
            field.setText(selected)

    def _set_configurations(self, plan: str) -> None:
        self.configuration.clear()
        self.configuration.addItems(configuration for option_plan, configuration in self._options if option_plan == plan)
        index = self.configuration.findText("3d_fullres")
        self.configuration.setCurrentIndex(max(index, 0))

    def command(self) -> str:
        return prediction_command(self.dataset_id, self.input_directory.text().strip(),
                                  self.output_directory.text().strip(), self.plan.currentText(),
                                  self.configuration.currentText(), self.additional_arguments.text())

    def _copy_command(self) -> None:
        if not self.input_directory.text().strip() or not self.output_directory.text().strip():
            QMessageBox.warning(self, "Directories required", "Choose both an input and output directory first.")
            return
        try:
            command = self.command()
        except ValueError as error:
            QToolTip.showText(self.additional_arguments.mapToGlobal(self.additional_arguments.rect().bottomLeft()),
                              f"Invalid arguments: {error}", self.additional_arguments)
            return
        QApplication.clipboard().setText(command)
        print(command)
        QToolTip.showText(self.copy.mapToGlobal(self.copy.rect().bottomLeft()), "Copied to clipboard", self.copy)

    def accept(self) -> None:
        if not self.input_directory.text().strip() or not self.output_directory.text().strip():
            QMessageBox.warning(self, "Directories required", "Choose both an input and output directory.")
            return
        try:
            self.command()
        except ValueError as error:
            QToolTip.showText(self.additional_arguments.mapToGlobal(self.additional_arguments.rect().bottomLeft()),
                              f"Invalid arguments: {error}", self.additional_arguments)
            return
        super().accept()
