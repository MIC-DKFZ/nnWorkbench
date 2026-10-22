"""Run file-system work outside the Qt UI thread."""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    """One per TaskRunner, in the UI thread: workers report through it (a queued connection to the runner)."""

    completed = Signal(int, object)  # task number, result
    failed = Signal(int, str)  # task number, error


class _Worker(QRunnable):
    def __init__(self, task: Callable[[], object], signals: _Signals, number: int) -> None:
        super().__init__()
        self.task = task
        self.signals = signals
        self.number = number

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
            getattr(self.signals, name).emit(self.number, value)
        except RuntimeError:
            pass  # the runner was deleted while the task ran (its window closed)


class TaskRunner(QObject):
    """A small thread pool; results are dropped when a newer request of the same key exists.

    Results reach the callbacks through this object's slots: once it is deleted (its window closed), Qt drops
    results that arrive later instead of calling into pages that no longer exist; after shutdown() they are
    ignored as well.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(min(4, max(2, (os.cpu_count() or 2) // 2)))
        self._versions: dict[str, int] = {}
        self._signals = _Signals(self)
        self._signals.completed.connect(self._completed)
        self._signals.failed.connect(self._failed)
        self._numbers = itertools.count()
        # task number → (key, version, done, failed) while the task runs
        self._workers: dict[int, tuple[str, int, Callable, Callable | None]] = {}

    def run(self, key: str, task: Callable[[], object], done: Callable[[object], None],
            failed: Callable[[str], None] | None = None) -> None:
        version = self._versions.get(key, 0) + 1
        self._versions[key] = version
        number = next(self._numbers)
        self._workers[number] = (key, version, done, failed)
        self._pool.start(_Worker(task, self._signals, number))

    def _completed(self, number: int, result: object) -> None:
        entry = self._workers.pop(number, None)
        if entry is not None and self._versions.get(entry[0]) == entry[1]:
            entry[2](result)

    def _failed(self, number: int, error: str) -> None:
        entry = self._workers.pop(number, None)
        if entry is not None and self._versions.get(entry[0]) == entry[1]:
            key, failed = entry[0], entry[3]
            (failed or (lambda message: print(f"Background task {key} failed: {message}")))(error)

    def shutdown(self) -> None:
        """No result is handled any more (the window closes); running tasks finish on their own."""
        self._versions.clear()
        self._pool.clear()

    @property
    def active(self) -> bool:
        """A task still runs (e.g. reading a network share that does not answer)."""
        return self._pool.activeThreadCount() > 0
