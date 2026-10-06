"""nnWorkbench workflow utilities."""

import importlib.util
import os

__version__ = "0.1.0"


def _prefer_pyside6() -> None:
    """Let qtpy use PySide6 when it is installed and no Qt binding was chosen explicitly.

    qtpy would otherwise prefer PyQt5, e.g. when napari was installed with it first. The setting is
    inherited by the Napari processes nnWorkbench starts, so all windows use the same binding.
    """
    if "QT_API" not in os.environ and importlib.util.find_spec("PySide6") is not None:
        os.environ["QT_API"] = "pyside6"


_prefer_pyside6()
