from __future__ import annotations

import signal
import sys

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .ui.style import apply_application_style


def main() -> None:
    app = QApplication(sys.argv); app.setApplicationName("nnWorkbench")
    apply_application_style(app)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Let Python process terminal signals while Qt owns the main event loop.
    sigint_timer = QTimer(app); sigint_timer.timeout.connect(lambda: None); sigint_timer.start(200)
    window = MainWindow(); window.show()
    app.aboutToQuit.connect(window.workspace.shutdown)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
