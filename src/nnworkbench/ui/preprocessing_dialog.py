from __future__ import annotations

from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QToolTip

from nnworkbench.services.commands import preprocessing_command
from .editable_dropdown import EditableDropdownComboBox


class PreprocessingDialog(QDialog):
    """Collect the parameters required to start nnU-Net preprocessing."""

    def __init__(self, dataset_id: int, planners: tuple[str, ...] = ("ExperimentPlanner",), parent=None) -> None:
        super().__init__(parent)
        self.dataset_id = dataset_id
        self.setWindowTitle("Run preprocessing")
        layout = QFormLayout(self)
        self.configuration = EditableDropdownComboBox()
        self.configuration.addItems(("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"))
        self.configuration.setCurrentText("3d_fullres")
        self.planner = EditableDropdownComboBox()
        self.planner.addItems(planners)
        self.planner.setEditable(True)
        self.planner.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.planner.setCurrentText("ExperimentPlanner")
        self.planner.setToolTip("Type a custom planner name, or use the arrow on the right to choose an installed planner.")
        layout.addRow("Configuration", self.configuration)
        layout.addRow("Planner", self.planner)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Run preprocessing")
        self.copy = buttons.addButton("Copy command", QDialogButtonBox.ButtonRole.ActionRole)
        self.copy.clicked.connect(self._copy_command)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def command(self) -> str:
        return preprocessing_command(self.dataset_id, self.configuration.currentText(), self.planner.currentText())

    def _copy_command(self) -> None:
        command = self.command()
        QApplication.clipboard().setText(command)
        print(command)
        QToolTip.showText(self.copy.mapToGlobal(self.copy.rect().bottomLeft()), "Copied to clipboard", self.copy)
