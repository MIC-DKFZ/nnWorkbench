"""Shared test setup."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True, scope="session")
def _one_application():
    """One QApplication for the whole run. Tests keep theirs in a local variable: PyQt5 deletes an application
    without Python references, so it was deleted and created again after every test while widgets of earlier
    tests could still exist, which crashed now and then (segmentation fault in a later test)."""
    from qtpy.QtCore import QStandardPaths
    from qtpy.QtWidgets import QApplication

    QStandardPaths.setTestModeEnabled(True)
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(autouse=True)
def _no_nnunet_folder_variables(monkeypatch):
    """nnWorkbench uses nnU-Net's folder variables when no paths are saved: tests must not see the developer's."""
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "nnUNet_extTrainer"):
        monkeypatch.delenv(name, raising=False)
