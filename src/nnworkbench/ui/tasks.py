"""Run file-system work outside the Qt UI thread."""

from __future__ import annotations

import os
from collections.abc import Callable

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class _Worker(QRunnable):
    def __init__(self, task: Callable[[], object]) -> None:
        super().__init__()
        self.task = task
        self.signals = _Signals()

    def run(self) -> None:
        try:
            result = self.task()
        # also SystemExit and KeyboardInterrupt: the caller must hear back, or its page waits forever
        except BaseException as error:  # noqa: BLE001 - reported to the caller instead of crashing the thread
            self._emit("failed", f"{type(error).__name__}: {error}")
        else:
            self._emit("completed", result)

    def _emit(self, name: str, value: object) -> None:
        try:
            getattr(self.signals, name).emit(value)
        except RuntimeError:
            pass  # the receiver was deleted while the task ran


class TaskRunner(QObject):
    """A small thread pool; results are dropped when a newer request of the same key exists."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(min(4, max(2, (os.cpu_count() or 2) // 2)))
        self._versions: dict[str, int] = {}
        self._workers: set[_Worker] = set()

    def run(self, key: str, task: Callable[[], object], done: Callable[[object], None],
            failed: Callable[[str], None] | None = None) -> None:
        version = self._versions.get(key, 0) + 1
        self._versions[key] = version
        worker = _Worker(task)
        self._workers.add(worker)

        def finish(result: object) -> None:
            self._workers.discard(worker)
            if self._versions.get(key) == version:
                done(result)

        def fail(error: str) -> None:
            self._workers.discard(worker)
            if self._versions.get(key) == version:
                (failed or (lambda message: print(f"Background task {key} failed: {message}")))(error)

        worker.signals.completed.connect(finish)
        worker.signals.failed.connect(fail)
        self._pool.start(worker)

    def shutdown(self) -> None:
        self._versions.clear()
        self._pool.clear()
