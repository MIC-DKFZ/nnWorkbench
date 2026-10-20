"""View and edit dataset-description.md (rendered Markdown with images, or plain text with Save)."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QStackedWidget,
    QVBoxLayout,
)

from nnworkbench.services.dataset_description import IMAGES_FOLDER, add_image

from . import style
from .widgets import button, label
from .workflow_guide import GuideBrowser

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.gif *.svg *.bmp *.webp)"


class DescriptionDialog(QDialog):
    """Shows the description like the guide; Edit switches to the Markdown text."""

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self.setWindowTitle(f"{path.parent.name} · {path.name}")
        self.resize(900, 820)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(label(path.name, name="h3"))
        self._state = label("", muted=True, size=12)
        head.addWidget(self._state)
        head.addStretch()
        self._add_image = button("Add image…", small=True, icon="file", clicked=self._insert_image,
                                 tooltip=f"Copy an image into {IMAGES_FOLDER}/ and insert it at the cursor")
        self._edit = button("Edit", small=True, icon="edit", clicked=self._start_edit)
        self._cancel = button("Cancel", small=True, clicked=self._cancel_edit)
        self._save = button("Save", small=True, primary=True, clicked=self._save_edit)
        open_folder = button("Open folder", small=True, icon="folder", clicked=self._open_folder,
                             tooltip="The dataset folder with the description and its images")
        for widget in (self._add_image, self._edit, self._cancel, self._save, open_folder):
            head.addWidget(widget)
        layout.addLayout(head)
        self._views = QStackedWidget()
        self._viewer = GuideBrowser()
        self._viewer.setOpenExternalLinks(True)
        self._editor = QPlainTextEdit()
        self._editor.setStyleSheet(f'font-family: "{style.MONO}"; font-size: 12px;')
        self._editor.textChanged.connect(self._mark_changed)
        self._views.addWidget(self._viewer)
        self._views.addWidget(self._editor)
        layout.addWidget(self._views, 1)
        hint = label("Markdown: # heading, **bold**, *italic*, - list, | table |, ![caption](description_images/x.png). "
                     "The file is plain text and can also be edited in any editor.", muted=True, size=11, wrap=True)
        layout.addWidget(hint)
        self._hint = hint
        self._show_view()

    # ------------------------------------------------------------ modes

    def _show_view(self) -> None:
        self._viewer.load(self._path)
        self._views.setCurrentWidget(self._viewer)
        for widget, visible in ((self._edit, True), (self._cancel, False), (self._save, False),
                                (self._add_image, False), (self._hint, False)):
            widget.setVisible(visible)
        self._state.setText("")

    def _start_edit(self) -> None:
        self._editor.blockSignals(True)
        self._editor.setPlainText(self._path.read_text(encoding="utf-8"))
        self._editor.blockSignals(False)
        self._views.setCurrentWidget(self._editor)
        for widget, visible in ((self._edit, False), (self._cancel, True), (self._save, True),
                                (self._add_image, True), (self._hint, True)):
            widget.setVisible(visible)
        self._state.setText("editing")
        self._editor.setFocus()

    def _mark_changed(self) -> None:
        self._state.setText("editing · unsaved changes")

    def _cancel_edit(self) -> None:
        if self._has_changes() and QMessageBox.question(
                self, "Discard changes?", "Discard the unsaved changes?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Discard:
            return
        self._show_view()

    def _save_edit(self) -> None:
        self._path.write_text(self._editor.toPlainText(), encoding="utf-8")
        self._show_view()

    def _has_changes(self) -> bool:
        return (self._views.currentWidget() is self._editor
                and self._editor.toPlainText() != self._path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ images and folder

    def _insert_image(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(self, "Add image", str(Path.home()), IMAGE_FILTER)
        if not selected:
            return
        target = add_image(self._path.parent, Path(selected))
        caption = Path(selected).stem.replace("_", " ")
        self._editor.insertPlainText(f"\n![{caption}]({target})\n")

    def _open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path.parent.resolve())))

    def reject(self) -> None:  # Escape or the window's close button
        if self._has_changes() and QMessageBox.question(
                self, "Discard changes?", "Close without saving the changes?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Discard:
            return
        super().reject()
