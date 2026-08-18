from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
                               QWidget)


class PathsHeader(QWidget):
    paths_changed = Signal()
    annotation_documentation_requested = Signal()
    annotation_documentation_closed = Signal()
    annotation_enabled_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("pathsHeader")
        self._fields: dict[str, QLineEdit] = {}
        self._documentation_visible = False
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 10, 12, 10); layout.setHorizontalSpacing(10); layout.setVerticalSpacing(8)
        for row, (key, title) in enumerate((("raw", "nnUNet_raw"), ("preprocessed", "nnUNet_preprocessed"),
                                             ("results", "nnUNet_results"))):
            field = QLineEdit()
            field.setPlaceholderText(f"Optional {title} directory")
            field.editingFinished.connect(self.paths_changed)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _=False, item=key: self._browse(item))
            layout.addWidget(QLabel(title), row, 0); layout.addWidget(field, row, 1); layout.addWidget(browse, row, 2)
            self._fields[key] = field

        self.annotation_group = QWidget()
        self.annotation_group.setObjectName("annotationWorkflow")
        self.annotation_group.setFixedWidth(150)
        annotation_layout = QVBoxLayout(self.annotation_group)
        annotation_layout.setContentsMargins(10, 9, 10, 9); annotation_layout.setSpacing(7)
        annotation_heading = QHBoxLayout(); annotation_heading.setSpacing(7)
        logo = QLabel()
        logo_icon = QIcon(str(Path(__file__).parent.parent / "assets" / "annotation-workflow.svg"))
        logo.setPixmap(logo_icon.pixmap(QSize(40, 40)))
        annotation_title = QLabel("Annotation\nworkflow")
        annotation_title.setObjectName("annotationWorkflowTitle")
        annotation_heading.addWidget(logo); annotation_heading.addWidget(annotation_title); annotation_heading.addStretch()
        annotation_layout.addLayout(annotation_heading)
        annotation_actions = QHBoxLayout(); annotation_actions.setSpacing(7)
        self.annotation = QPushButton("Off")
        self.annotation.setObjectName("annotationToggle")
        self.annotation.setCheckable(True)
        self.annotation.setToolTip("Enable the optional interactive annotation workflow")
        self.annotation.toggled.connect(self._set_annotation_state)
        self.annotation_documentation = QPushButton("ⓘ")
        self.annotation_documentation.setObjectName("annotationDocumentation")
        self.annotation_documentation.setToolTip("Show the annotation workflow documentation")
        self.annotation_documentation.clicked.connect(self._toggle_annotation_documentation)
        annotation_actions.addWidget(self.annotation); annotation_actions.addWidget(self.annotation_documentation)
        annotation_layout.addLayout(annotation_actions)
        self._set_annotation_state(False)
        layout.addWidget(self.annotation_group, 0, 3, 3, 1)

    def _set_annotation_state(self, active: bool) -> None:
        """Show the future annotation-workflow switch state without changing workflow behavior."""
        self.annotation.setText("On" if active else "Off")
        self.annotation_group.setProperty("active", active)
        self.annotation_group.style().unpolish(self.annotation_group); self.annotation_group.style().polish(self.annotation_group)
        self.annotation_enabled_changed.emit(active)

    def set_annotation_enabled(self, active: bool) -> None:
        if self.annotation.isChecked() != active:
            self.annotation.setChecked(active)
        else:
            self._set_annotation_state(active)

    def annotation_enabled(self) -> bool:
        return self.annotation.isChecked()

    def set_annotation_documentation_visible(self, visible: bool) -> None:
        self._documentation_visible = visible
        self.annotation_documentation.setToolTip(
            "Close the annotation workflow documentation" if visible else "Show the annotation workflow documentation")

    def _toggle_annotation_documentation(self) -> None:
        if self._documentation_visible:
            self.annotation_documentation_closed.emit()
        else:
            self.annotation_documentation_requested.emit()

    def _browse(self, key: str) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose nnU-Net directory", self._fields[key].text())
        if selected:
            self._fields[key].setText(selected); self.paths_changed.emit()

    def paths(self) -> tuple[Path | None, Path | None, Path | None]:
        return tuple(Path(field.text()).expanduser() if field.text().strip() else None
                     for field in self._fields.values())  # type: ignore[return-value]

    def set_paths(self, raw: str, preprocessed: str, results: str) -> None:
        for field, value in zip(self._fields.values(), (raw, preprocessed, results)):
            field.setText(value)
