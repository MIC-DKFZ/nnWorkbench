"""Minimal offscreen checks for the Qt application shell."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication

from nnworkbench.ui.main_window import MainWindow


def test_main_window_can_be_created_offscreen() -> None:
    QStandardPaths.setTestModeEnabled(True)
    application = QApplication.instance() or QApplication([])
    window = MainWindow()

    try:
        assert window.windowTitle().startswith("nnWorkbench")
        assert window.splitter.count() == 3
    finally:
        window.close()
        application.processEvents()
