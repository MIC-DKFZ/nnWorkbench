"""About nnWorkbench: version, what it builds on, the device nnU-Net uses, who develops it (with the logos) and the
licence."""

from __future__ import annotations

from html import escape
from importlib import metadata

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from nnworkbench.services.device_probe import PYTORCH_URL, DeviceInfo

from .widgets import Card, badge, button, icon_label, label, logo_label, separator, tip

_WIDTH = 560


def _version() -> str:
    try:
        return metadata.version("nnworkbench")
    except metadata.PackageNotFoundError:
        return "unknown"


def _link(url: str, text: str) -> str:
    return f"<a href='{url}' style='color:#1f5fbf; text-decoration:none'>{text}</a>"


class AboutDialog(QDialog):
    def __init__(self, parent=None, device: DeviceInfo | None = None) -> None:
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
        layout.addWidget(self._device_card(device))
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

    def _device_card(self, device: DeviceInfo | None) -> QWidget:
        """The device nnU-Net trains and predicts on (GPU, Apple GPU or CPU), and how to fix a missing GPU."""
        card = Card(padding=14)
        card.body.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(icon_label("chip", 16))
        head.addWidget(label("Device for training and prediction", name="h3"))
        head.addStretch()
        status, tone = device.status if device is not None else ("checking…", "gray")
        head.addWidget(badge(status, tone))
        card.body.addLayout(head)
        if device is None:
            card.body.addWidget(label("Checking which device PyTorch can use…", muted=True, size=12))
            return card
        rows = []
        if device.kind == "cuda":
            rows += [("GPU" if len(device.gpus) == 1 else f"GPU {index}", f"{name} · {memory:.0f} GB memory")
                     for index, (name, memory) in enumerate(device.gpus)]
        elif device.kind == "mps":
            rows.append(("GPU", "Apple GPU (MPS)"))
        elif device.kind == "cpu":
            rows.append(("GPU", "none usable, nnU-Net runs on the CPU"))
        if device.torch_version:
            rows.append(("PyTorch", device.torch_version + (f" · CUDA {device.cuda_build}" if device.cuda_build
                                                              else " · without CUDA")))
        if rows:
            grid = QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(4)
            for row, (name, value) in enumerate(rows):
                grid.addWidget(label(name, muted=True, size=12), row, 0)
                grid.addWidget(label(value, size=12), row, 1)
            grid.setColumnStretch(1, 1)
            card.body.addLayout(grid)
        if device.problem:
            text = escape(device.problem).replace("pytorch.org", _link(PYTORCH_URL, "pytorch.org"))
            card.body.addWidget(self._tip(text, "orange"))
            if device.fix_command:
                card.body.addLayout(self._command_row(device.fix_command))
        elif device.kind == "cpu":
            card.body.addWidget(self._tip(
                "Training on the CPU takes days per fold. Use a computer with a GPU or a cluster (see the remote "
                "installation in the README).", "gray"))
        return card

    @staticmethod
    def _tip(text: str, tone: str) -> QLabel:
        widget = tip(text, tone)
        widget.setOpenExternalLinks(True)
        # reserve the wrapped height: card margins and borders (2 × 15) and the tip's padding (2 × 11 and 2 × 9)
        widget.setMinimumHeight(widget.heightForWidth(_WIDTH - 48 - 30 - 22) + 18)
        return widget

    @staticmethod
    def _command_row(command: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        # one line break before the URL, so it is not split in the middle
        text = label(escape(command).replace(" https://", "<br>https://"), mono=True, rich=True, size=11)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(text, 1)
        row.addWidget(button("Copy", small=True, clicked=lambda: QGuiApplication.clipboard().setText(command)))
        return row

    @staticmethod
    def _text(text: str, muted: bool = False) -> QLabel:
        widget = label(text, wrap=True, rich=True, muted=muted, size=12 if muted else None)
        widget.setOpenExternalLinks(True)
        widget.setMinimumHeight(widget.heightForWidth(_WIDTH - 48))  # reserve the wrapped height (fixed width)
        return widget

