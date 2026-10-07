"""Settings: Annotation Workflow (global), analysis workers and Napari prefetching."""

from __future__ import annotations

import os

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QSpinBox,
    QVBoxLayout,
)

from .context import Settings
from .widgets import ToggleSwitch, button, label, section_label, separator


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(20, 16, 20, 16)
        head.addWidget(label("Settings", name="h2"))
        layout.addLayout(head)
        layout.addWidget(separator())

        def section(title: str, rows, hint: str) -> None:
            body = QVBoxLayout()
            body.setContentsMargins(20, 14, 20, 14)
            body.setSpacing(10)
            body.addWidget(section_label(title))
            grid = QGridLayout()
            grid.setHorizontalSpacing(16)
            grid.setVerticalSpacing(8)
            for row, (name, widget) in enumerate(rows):
                grid.addWidget(label(name), row, 0)
                grid.addWidget(widget, row, 1)
            grid.setColumnMinimumWidth(0, 200)
            grid.setColumnStretch(2, 1)
            body.addLayout(grid)
            body.addWidget(label(hint, muted=True, wrap=True, size=12))
            layout.addLayout(body)
            layout.addWidget(separator())

        self.annotation = ToggleSwitch("On")
        self.annotation.setChecked(settings.annotation_workflow)
        self.annotation.toggled.connect(lambda on: self.annotation.setText("On" if on else "Off"))
        self.annotation.setText("On" if settings.annotation_workflow else "Off")
        section("Workflow", [("Annotation Workflow", self.annotation)],
                "Applies to all datasets. Adds ✎ to annotate cases in Napari (saved directly to labelsTr) and the "
                "annotation hints. Also available as the switch in the top bar.")
        self.workers = QSpinBox()
        self.workers.setRange(1, os.cpu_count() or 1)
        self.workers.setValue(min(settings.workers, os.cpu_count() or 1))
        section("Label analysis", [("Worker processes", self.workers)], "Used by Analyze dataset.")
        self.previous = QCheckBox("Previous cases")
        self.previous.setChecked(settings.prefetch_previous)
        self.following = QCheckBox("Next cases")
        self.following.setChecked(settings.prefetch_next)
        prefetch = QFrame()
        prefetch_row = QHBoxLayout(prefetch)
        prefetch_row.setContentsMargins(0, 0, 0, 0)
        prefetch_row.addWidget(self.previous)
        prefetch_row.addWidget(self.following)
        self.radius = QSpinBox()
        self.radius.setRange(1, 20)
        self.radius.setValue(settings.prefetch_radius)
        section("Napari", [("Prefetch", prefetch), ("Prefetch radius", self.radius)],
                "Loads neighbouring cases in the background so stepping through cases in Napari is faster. Uses more "
                "memory.")
        folders = QVBoxLayout()
        folders.setContentsMargins(20, 14, 20, 14)
        folders.addWidget(section_label("nnU-Net folders"))
        folders.addWidget(label("The nnUNet_raw, nnUNet_preprocessed and nnUNet_results paths are set on the overview "
                                "(← All datasets).", muted=True, wrap=True, size=12))
        layout.addLayout(folders)
        actions = QHBoxLayout()
        actions.setContentsMargins(20, 12, 20, 16)
        actions.addStretch()
        actions.addWidget(button("Cancel", clicked=self.reject))
        actions.addWidget(button("Save", primary=True, clicked=self.accept))
        layout.addLayout(actions)

    def settings(self) -> Settings:
        return Settings(self.annotation.isChecked(), self.workers.value(), self.previous.isChecked(),
                        self.following.isChecked(), self.radius.value())
