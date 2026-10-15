"""About nnWorkbench: version, what it builds on, who develops it (with the logos) and the licence."""

from __future__ import annotations

from importlib import metadata

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout

from .widgets import button, label, logo_label, separator

_WIDTH = 560


def _version() -> str:
    try:
        return metadata.version("nnworkbench")
    except metadata.PackageNotFoundError:
        return "unknown"


def _link(url: str, text: str) -> str:
    return f"<a href='{url}' style='color:#1f5fbf; text-decoration:none'>{text}</a>"


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About nnWorkbench")
        self.setFixedWidth(_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(12)
        head = QHBoxLayout()
        head.setSpacing(12)
        mark = QLabel("nW")
        mark.setFixedSize(40, 40)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1f5fbf, stop:1 #5b8fe0);"
                           "color: white; font-weight: 800; font-size: 16px; border-radius: 10px;")
        head.addWidget(mark)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(label("nnWorkbench", name="h2"))
        titles.addWidget(label(f"Version {_version()}", muted=True, size=12))
        head.addLayout(titles, 1)
        layout.addLayout(head)
        layout.addWidget(self._text("A GUI for nnU-Net workflows and nnInteractive-powered annotation: from the first "
                                    "image to a trained segmentation model in one place."))
        layout.addWidget(self._text(
            f"Built on {_link('https://github.com/MIC-DKFZ/nnUNet', 'nnU-Net')} and "
            f"{_link('https://github.com/MIC-DKFZ/napari-nninteractive', 'nnInteractive')}, with Napari for "
            "viewing and annotating images."))
        layout.addWidget(separator())
        logos = QHBoxLayout()
        logos.setSpacing(32)
        logos.addStretch()
        logos.addWidget(logo_label("helmholtz-imaging-logo", 38))
        logos.addWidget(logo_label("dkfz-logo", 30))
        logos.addStretch()
        layout.addSpacing(6)
        layout.addLayout(logos)
        layout.addSpacing(8)
        layout.addWidget(self._text(
            "Developed and maintained by the Applied Computer Vision Lab (ACVL) of "
            f"{_link('https://www.helmholtz-imaging.de/', 'Helmholtz Imaging')} at the German Cancer Research "
            "Center (DKFZ)."))
        layout.addWidget(separator())
        licences = self._text(
            "nnWorkbench is licensed under the Apache License 2.0.<br>"
            f"The {_link('https://huggingface.co/MIC-DKFZ/nnInteractive', 'nnInteractive model weights')}, downloaded "
            f"on first use, are licensed under "
            f"{_link('https://creativecommons.org/licenses/by-nc-sa/4.0/', 'CC&nbsp;BY-NC-SA&nbsp;4.0')}: "
            "<b>non-commercial use only</b>.", muted=True)
        layout.addWidget(licences)
        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(button("Close", clicked=self.accept))
        layout.addLayout(footer)

    @staticmethod
    def _text(text: str, muted: bool = False) -> QLabel:
        widget = label(text, wrap=True, rich=True, muted=muted, size=12 if muted else None)
        widget.setOpenExternalLinks(True)
        widget.setMinimumHeight(widget.heightForWidth(_WIDTH - 48))  # reserve the wrapped height (fixed width)
        return widget
