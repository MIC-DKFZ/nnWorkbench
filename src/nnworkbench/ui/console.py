"""Collapsible console at the bottom of every screen."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication, QTextCursor
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from .jobs import Job, JobRunner
from .widgets import button, clear_layout, label, polish

# Output without line breaks (dots, a progress bar without "\r") would grow one text block without end, and every
# insertion lays out the whole block again: the console breaks such lines after this many characters.
LINE_LIMIT = 1_000
MAX_BLOCKS = 20_000  # lines kept in the console
_SYMBOLS = {"running": "●", "finished": "✓", "failed": "✗", "terminated": "■"}


def break_long_lines(text: str, column: int = 0, limit: int = LINE_LIMIT) -> str:
    """Insert line breaks so no line is longer than *limit*; *column* characters are already on the first line."""
    if column + len(text) <= limit:
        return text
    lines = text.split("\n")
    for index, line in enumerate(lines):
        room = max(0, limit - (column if index == 0 else 0))
        if len(line) > room:
            lines[index] = "\n".join([line[:room], *(line[start:start + limit]
                                                    for start in range(room, len(line), limit))])
    return "\n".join(lines)


class ConsolePanel(QFrame):
    """One line when collapsed; the full log of the selected job when expanded."""

    def __init__(self, runner: JobRunner, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("footer")
        self._runner = runner
        self._selected: Job | None = None
        self._expanded = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QWidget()
        header.setFixedHeight(40)
        row = QHBoxLayout(header)
        row.setContentsMargins(24, 0, 24, 0)
        row.setSpacing(10)
        row.addWidget(label("Console", bold=True))
        self._chips = QHBoxLayout()
        self._chips.setSpacing(6)
        row.addLayout(self._chips)
        self._progress = label("", bold=True, size=12)
        self._progress.setStyleSheet("color: #1f5fbf;")
        row.addWidget(self._progress)
        self._last_line = label("No commands have been run in this session.", muted=True, mono=True)
        self._last_line.setMinimumWidth(50)
        row.addWidget(self._last_line, 1)
        self._spacer = QWidget()
        row.addWidget(self._spacer, 1)
        self._summary = label("", muted=True, size=12)
        row.addWidget(self._summary)
        self._copy = button("Copy output", small=True, clicked=self._copy_output)
        self._terminate = button("■ Terminate", small=True, danger=True, clicked=runner.terminate)
        self._toggle = button("▴ Expand", ghost=True, small=True, clicked=self.toggle)
        for widget in (self._copy, self._terminate, self._toggle):
            row.addWidget(widget)
        layout.addWidget(header)
        self._log = QPlainTextEdit()
        self._log.setObjectName("consoleLog")
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(MAX_BLOCKS)
        self._log.setPlaceholderText("No commands have been run in this session.")
        log_holder = QWidget()
        holder_layout = QVBoxLayout(log_holder)
        holder_layout.setContentsMargins(24, 0, 24, 12)
        holder_layout.addWidget(self._log)
        self._log_holder = log_holder
        layout.addWidget(log_holder)
        runner.job_started.connect(self._job_started)
        runner.job_output.connect(self._job_output)
        runner.job_finished.connect(lambda _job: self._refresh())
        self._apply_expanded()
        self._refresh()

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._apply_expanded()

    def _apply_expanded(self) -> None:
        self._log_holder.setVisible(self._expanded)
        self._toggle.setText("▾ Collapse" if self._expanded else "▴ Expand")
        self._copy.setVisible(self._expanded)
        self._last_line.setVisible(not self._expanded)
        self._spacer.setVisible(self._expanded)
        self._summary.setVisible(not self._expanded)
        self.setFixedHeight(300 if self._expanded else 40)
        self._refresh()

    def _job_started(self, job: Job) -> None:
        self._selected = job
        self._show_job(job)
        self._refresh()

    def _job_output(self, job: Job, text: str) -> None:
        if job is self._selected:
            cursor = self._log.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(break_long_lines(text, cursor.positionInBlock()))
            self._log.setTextCursor(cursor)
            self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
        if job is self._runner.current:
            self._last_line.setText(job.last_line)
            self._progress.setText(job.progress)
            self._progress.setVisible(bool(job.progress))

    def _show_job(self, job: Job) -> None:
        # job.output holds at most OUTPUT_LIMIT characters (jobs.py); show only the lines the console keeps
        lines = break_long_lines("".join(job.output)).split("\n")
        if len(lines) > MAX_BLOCKS:
            notice = f"[Earlier output is not shown here; the full output is in the log file: {job.log}]"
            lines = [notice, *lines[-(MAX_BLOCKS - 1):]]
        self._log.setPlainText("\n".join(lines))
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _select(self, job: Job) -> None:
        self._selected = job
        self._show_job(job)
        self._refresh()

    def _refresh(self) -> None:
        clear_layout(self._chips)
        jobs = self._runner.jobs
        current = self._runner.current
        shown = jobs[-4:] if self._expanded else ([current] if current else jobs[-1:])
        for job in shown:
            symbol = _SYMBOLS.get(job.status, "●")  # e.g. "running in the background"
            suffix = f" · {job.started:%H:%M}" if job.status != "running" else " · running"
            chip = QLabel(f"{symbol} {job.label}{suffix if self._expanded else ''}")
            chip.setObjectName("jobChip")
            chip.setFixedHeight(24)
            chip.setProperty("active", job is (self._selected if self._expanded else current or job))
            chip.setProperty("failed", job.status == "failed")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.mousePressEvent = lambda _event, selected=job: self._select(selected)  # type: ignore[assignment]
            polish(chip)
            self._chips.addWidget(chip)
        earlier = max(len(jobs) - 1, 0)  # the bar shows one job (the running or the last one)
        self._summary.setText(f"{earlier} earlier job{'s' if earlier != 1 else ''}" if earlier else "")
        if current:
            self._last_line.setText(current.last_line or current.command)
        elif jobs:
            last = jobs[-1]
            if last.status == "failed":
                code = f" (exit code {last.exit_code})" if last.exit_code is not None else ""
                self._last_line.setText(f"✗ failed{code} · {last.last_line or 'expand to see the output'}")
            else:
                self._last_line.setText(f"{last.label} {last.status}")
        self._last_line.setProperty("failed", bool(jobs and not current and jobs[-1].status == "failed"))
        polish(self._last_line)
        self._terminate.setVisible(current is not None)
        self._progress.setText(current.progress if current else "")
        self._progress.setVisible(bool(current and current.progress))

    def _copy_output(self) -> None:
        QGuiApplication.clipboard().setText(self._log.toPlainText())
