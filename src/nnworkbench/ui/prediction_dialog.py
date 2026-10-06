from __future__ import annotations

from qtpy.QtWidgets import (
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

from nnworkbench.services.commands import prediction_command, wrapped_command

from .editable_dropdown import EditableDropdownComboBox


class PredictionDialog(QDialog):
    """Collect parameters required to run prediction from an available trained model."""

    def __init__(self, dataset_id: int, trained_models: list[tuple[str, str, str]],
                 local_run_available: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.dataset_id = dataset_id
        self.setWindowTitle("Run prediction")
        self._options = sorted(set(trained_models))
        layout = QFormLayout(self)
        self.input_directory = QLineEdit(); self.output_directory = QLineEdit()
        self.plan = EditableDropdownComboBox(); self.plan.addItems(sorted({plan for _, plan, _ in self._options}))
        self.plan.setEditable(True)
        self.trainer = EditableDropdownComboBox()
        self.trainer.addItems(sorted({trainer for trainer, _, _ in self._options}))
        self.trainer.setEditable(True)
        self.configuration = EditableDropdownComboBox()
        self.configuration.setEditable(True)
        self.folds = QLineEdit("0 1 2 3 4")
        self.folds.setPlaceholderText("0 1 2 3 4")
        self.folds.setToolTip("Space-separated folds used for prediction. nnU-Net defaults to 0 1 2 3 4.")
        self.additional_arguments = QLineEdit()
        self.additional_arguments.setPlaceholderText("Optional, e.g. --save_probabilities")
        self.additional_arguments.setToolTip("Extra arguments passed directly to nnUNetv2_predict.")
        self.plan.currentTextChanged.connect(self._set_configurations)
        self.trainer.currentTextChanged.connect(self._set_configurations)
        self._set_configurations()
        layout.addRow("Input directory", self._directory_field(self.input_directory, "Choose input directory"))
        layout.addRow("Output directory", self._directory_field(self.output_directory, "Choose output directory"))
        layout.addRow("Plans", self.plan)
        layout.addRow("Configuration", self.configuration)
        layout.addRow("Trainer", self.trainer)
        layout.addRow("Folds", self.folds)
        layout.addRow("Additional arguments", self.additional_arguments)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        run = buttons.button(QDialogButtonBox.StandardButton.Ok)
        run.setText("Run prediction")
        run.setEnabled(local_run_available)
        if not local_run_available:
            run.setToolTip("A local nnUNet_results directory is required to run here.")
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

    def _set_configurations(self, *_unused: str) -> None:
        self.configuration.clear()
        self.configuration.addItems(
            configuration for trainer, plan, configuration in self._options
            if trainer == self.trainer.currentText() and plan == self.plan.currentText()
        )
        index = self.configuration.findText("3d_fullres")
        self.configuration.setCurrentIndex(max(index, 0))

    def command(self) -> str:
        return prediction_command(self.dataset_id, self.input_directory.text().strip(),
                                  self.output_directory.text().strip(), self.plan.currentText(),
                                  self.configuration.currentText(), self.additional_arguments.text(),
                                  trainer=self.trainer.currentText(), folds=self.folds.text())

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
        command = wrapped_command(command)
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
