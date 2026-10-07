"""Collapsible console at the bottom of every screen."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication, QTextCursor
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from .jobs import Job, JobRunner
from .widgets import button, clear_layout, label, polish


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
        self._log.setMaximumBlockCount(20_000)
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

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
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
            cursor.insertText(text)
            self._log.setTextCursor(cursor)
            self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
        if job is self._runner.current:
            self._last_line.setText(job.last_line)

    def _show_job(self, job: Job) -> None:
        self._log.setPlainText("".join(job.output))
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
            symbol = {"running": "●", "finished": "✓", "failed": "✗", "terminated": "■"}[job.status]
            suffix = f" · {job.started:%H:%M}" if job.status != "running" else " · running"
            chip = QLabel(f"{symbol} {job.label}{suffix if self._expanded else ''}")
            chip.setObjectName("jobChip")
            chip.setFixedHeight(24)
            chip.setProperty("active", job is (self._selected if self._expanded else current or job))
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.mousePressEvent = lambda _event, selected=job: self._select(selected)  # type: ignore[assignment]
            polish(chip)
            self._chips.addWidget(chip)
        earlier = len(jobs) - (1 if current else 0)
        self._summary.setText(f"{earlier} earlier job{'s' if earlier != 1 else ''}" if jobs else "")
        if current:
            self._last_line.setText(current.last_line or current.command)
        elif jobs:
            self._last_line.setText(f"{jobs[-1].label} {jobs[-1].status}")
        self._terminate.setVisible(current is not None)

    def _copy_output(self) -> None:
        QGuiApplication.clipboard().setText(self._log.toPlainText())
