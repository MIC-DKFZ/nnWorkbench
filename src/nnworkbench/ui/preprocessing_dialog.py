from __future__ import annotations

from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QToolTip,
)

from nnworkbench.services.commands import preprocessing_command, wrapped_command

from .editable_dropdown import EditableDropdownComboBox

DEFAULT_PLANNER = "nnUNetPlannerResEncL"


class PreprocessingDialog(QDialog):
    """Collect the parameters required to start nnU-Net preprocessing."""

    def __init__(self, dataset_id: int, planners: tuple[str, ...] = ("ExperimentPlanner",),
                 local_run_available: bool = True, parent=None) -> None:
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
        self.planner.setCurrentText(DEFAULT_PLANNER)
        self.planner.setToolTip("Type a custom planner name, or use the arrow on the right to choose an installed planner.")
        self.additional_arguments = QLineEdit()
        self.additional_arguments.setPlaceholderText("Optional, e.g. --np 8")
        self.additional_arguments.setToolTip("Extra arguments passed directly to nnUNetv2_plan_and_preprocess.")
        layout.addRow("Configuration", self.configuration)
        layout.addRow("Planner", self.planner)
        layout.addRow("Additional arguments", self.additional_arguments)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        run = buttons.button(QDialogButtonBox.StandardButton.Ok)
        run.setText("Run preprocessing")
        run.setEnabled(local_run_available)
        if not local_run_available:
            run.setToolTip("Local nnUNet_raw and nnUNet_preprocessed directories are required to run here.")
        self.copy = buttons.addButton("Copy command", QDialogButtonBox.ButtonRole.ActionRole)
        self.copy.clicked.connect(self._copy_command)
        buttons.accepted.connect(self._accept_if_valid); buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def command(self) -> str:
        return preprocessing_command(self.dataset_id, self.configuration.currentText(), self.planner.currentText(),
                                     self.additional_arguments.text())

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
        command = wrapped_command(command)
        QApplication.clipboard().setText(command)
        print(command)
        QToolTip.showText(self.copy.mapToGlobal(self.copy.rect().bottomLeft()), "Copied to clipboard", self.copy)
