from __future__ import annotations

import importlib.util
import os
import signal
import sys

# The core installation (`pip install nnworkbench`) has nnU-Net and the command-line tools only; the app needs these.
GUI_PACKAGES = {"qtpy": "qtpy", "a Qt binding": ("PySide6", "PyQt5"), "napari": "napari", "mitk-python": "mitk"}
INSTALL_ALL = 'pip install "nnworkbench[all]"'


def missing_gui_packages() -> list[str]:
    """The GUI packages that are not installed (empty for the full installation)."""
    def installed(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    return [name for name, modules in GUI_PACKAGES.items()
            if not any(installed(module) for module in ((modules,) if isinstance(modules, str) else modules))]


def missing_gui_message(missing: list[str]) -> str:
    names = missing[0] if len(missing) == 1 else f"{', '.join(missing[:-1])} and {missing[-1]}"
    return (f"nnworkbench-gui needs the full installation, but {names} "
            f"{'is' if len(missing) == 1 else 'are'} not installed. This environment has the core installation only "
            "(nnU-Net and the command-line tools, e.g. for a cluster).\n\n"
            f"Install the app with:\n    {INSTALL_ALL}\n"
            '(from a source checkout: pip install -e ".[all]")')


def main() -> None:
    missing = missing_gui_packages()
    if missing:
        raise SystemExit(missing_gui_message(missing))
    _run()


def _run() -> None:
    from qtpy import QT5
    from qtpy.QtCore import Qt, QTimer
    from qtpy.QtWidgets import QApplication

    from .ui.main_window import MainWindow
    from .ui.style import apply_application_style

    if QT5:  # Qt 6 scales for high-DPI screens by default; Qt 5 needs this before the application exists
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv); app.setApplicationName("nnWorkbench")
    apply_application_style(app)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Let Python process terminal signals while Qt owns the main event loop.
    sigint_timer = QTimer(app); sigint_timer.timeout.connect(lambda: None); sigint_timer.start(200)
    window = MainWindow(); window.show()
    window.context.detect_device()  # in the background; the app bar shows the result
    app.aboutToQuit.connect(window.context.shutdown)
    code = app.exec()
    if window.context.tasks.active:
        # A read still waits for a slow or disconnected network share: the thread pool would keep the (invisible)
        # process alive until it answers. Background tasks only read, so nothing is lost by ending here.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(code)


if __name__ == "__main__":
    main()
