from __future__ import annotations

from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QToolTip

from nnworkbench.services.commands import training_command
from .editable_dropdown import EditableDropdownComboBox


class TrainingDialog(QDialog):
    """Collect parameters required to start an nnU-Net training run."""

    def __init__(self, dataset_id: int, plans: list[tuple[str, tuple[str, ...]]],
                 trainers: tuple[str, ...] = ("nnUNetTrainer",), parent=None) -> None:
        super().__init__(parent)
        self.dataset_id = dataset_id
        self.setWindowTitle("Run training")
        self._plans = dict(plans)
        layout = QFormLayout(self)
        self.plan_file = EditableDropdownComboBox(); self.plan_file.addItems(name for name, _ in plans)
        self.configuration = EditableDropdownComboBox()
        self.trainer = EditableDropdownComboBox(); self.trainer.addItems(trainers); self.trainer.setEditable(True)
        self.trainer.setInsertPolicy(QComboBox.InsertPolicy.NoInsert); self.trainer.setCurrentText("nnUNetTrainer")
        self.trainer.setToolTip("Type a custom trainer name, or use the arrow on the right to choose an installed trainer.")
        self.fold = QLineEdit("0"); self.fold.setPlaceholderText("0, 1, …, or all")
        self.additional_arguments = QLineEdit()
        self.additional_arguments.setPlaceholderText("Optional, e.g. --npz")
        self.additional_arguments.setToolTip("Extra arguments passed directly to nnUNetv2_train.")
        self.plan_file.currentTextChanged.connect(self._set_configurations)
        self._set_configurations(self.plan_file.currentText())
        layout.addRow("Plans file", self.plan_file)
        layout.addRow("Configuration", self.configuration)
        layout.addRow("Trainer", self.trainer)
        layout.addRow("Fold", self.fold)
        layout.addRow("Additional arguments", self.additional_arguments)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Run training")
        self.copy = buttons.addButton("Copy command", QDialogButtonBox.ButtonRole.ActionRole)
        self.copy.clicked.connect(self._copy_command)
        buttons.accepted.connect(self._accept_if_valid); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _set_configurations(self, plan_name: str) -> None:
        self.configuration.clear()
        self.configuration.addItems(self._plans.get(plan_name, ()))
        index = self.configuration.findText("3d_fullres")
        self.configuration.setCurrentIndex(index if index >= 0 else 0)

    def command(self) -> str:
        return training_command(self.dataset_id, self.plan_file.currentText(), self.configuration.currentText(),
                                self.trainer.currentText(), self.fold.text().strip(), self.additional_arguments.text())

    def _accept_if_valid(self) -> None:
        try:
            self.command()
        except ValueError as error:
            QToolTip.showText(self.additional_arguments.mapToGlobal(self.additional_arguments.rect().bottomLeft()),
                              f"Invalid arguments: {error}", self.additional_arguments)
            return
        self.accept()

    def _copy_command(self) -> None:
        try:
            command = self.command()
        except ValueError as error:
            QToolTip.showText(self.additional_arguments.mapToGlobal(self.additional_arguments.rect().bottomLeft()),
                              f"Invalid arguments: {error}", self.additional_arguments)
            return
        QApplication.clipboard().setText(command)
        print(command)
        QToolTip.showText(self.copy.mapToGlobal(self.copy.rect().bottomLeft()), "Copied to clipboard", self.copy)
