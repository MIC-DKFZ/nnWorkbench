"""Settings: analysis workers and Napari prefetching (the Annotation Workflow is switched in the top bar)."""

from __future__ import annotations

import os

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLayout,
    QVBoxLayout,
)

from .context import Settings
from .widgets import NumberField, button, label, section_label, separator, with_stepper


def _number_field() -> NumberField:
    field = NumberField(1, 99, 1)
    field.setFixedWidth(100)
    field.setMinimumHeight(32)
    return field


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedWidth(620)  # a fixed width lets the wrapped hints compute their height correctly
        layout = QVBoxLayout(self)
        # Grow with the wrapped hint texts instead of letting them overlap the next section.
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
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
            grid.setVerticalSpacing(12)
            for row, (name, widget) in enumerate(rows):
                grid.addWidget(label(name), row, 0)
                grid.addWidget(widget, row, 1, Qt.AlignmentFlag.AlignLeft)
                grid.setRowMinimumHeight(row, 34)
            grid.setColumnMinimumWidth(0, 200)
            grid.setColumnStretch(2, 1)
            body.addLayout(grid)
            text = label(hint, muted=True, wrap=True, size=12)
            text.setMinimumHeight(text.heightForWidth(420))  # reserve the wrapped height (generously), so nothing overlaps
            body.addWidget(text)
            layout.addLayout(body)
            layout.addWidget(separator())

        self._annotation = settings.annotation_workflow  # switched in the top bar, kept unchanged here
        self.workers = _number_field()
        self.workers.setRange(1, os.cpu_count() or 1)
        self.workers.setValue(min(settings.workers, os.cpu_count() or 1))
        section("Statistics", [("Worker processes", with_stepper(self.workers))], "Used by Compute statistics.")
        self.previous = QCheckBox("Previous cases")
        self.previous.setChecked(settings.prefetch_previous)
        self.following = QCheckBox("Next cases")
        self.following.setChecked(settings.prefetch_next)
        prefetch = QFrame()
        prefetch_row = QHBoxLayout(prefetch)
        prefetch_row.setContentsMargins(0, 0, 0, 0)
        prefetch_row.addWidget(self.previous)
        prefetch_row.addWidget(self.following)
        self.radius = _number_field()
        self.radius.setRange(1, 20)
        self.radius.setValue(settings.prefetch_radius)
        section("Napari", [("Prefetch", prefetch), ("Prefetch radius", with_stepper(self.radius))],
                "Loads neighbouring cases in the background so stepping through cases in Napari is faster. Uses more "
                "memory.")
        actions = QHBoxLayout()
        actions.setContentsMargins(20, 12, 20, 16)
        actions.addStretch()
        actions.addWidget(button("Cancel", clicked=self.reject))
        actions.addWidget(button("Save", primary=True, clicked=self.accept))
        layout.addLayout(actions)

    def settings(self) -> Settings:
        return Settings(self._annotation, self.workers.value(), self.previous.isChecked(),
                        self.following.isChecked(), self.radius.value())
