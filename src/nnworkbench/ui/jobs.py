"""Run one command at a time (analysis, preprocessing, training, prediction) and keep its output."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from qtpy.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal


@dataclass
class Job:
    title: str  # e.g. "Train · fold 2"
    dataset: str  # e.g. "MarrowCells"
    command: str
    started: datetime = field(default_factory=datetime.now)
    status: str = "running"  # running, finished, failed, terminated
    output: list[str] = field(default_factory=list)
    last_line: str = ""

    @property
    def label(self) -> str:
        return f"{self.dataset} · {self.title}" if self.dataset else self.title


def start_process_group(process: QProcess, program: str, arguments: list[str]) -> None:
    """Launch a child in its own process group, so its worker processes can be stopped too."""
    process.setProperty("isolated_process_group", False)
    if os.name == "posix":
        process.setProperty("isolated_process_group", True)
        process.start(sys.executable, ["-m", "nnworkbench.services.process_runner", program, *arguments])
    else:
        process.start(program, arguments)


def stop_process_group(process: QProcess, wait_ms: int = 0) -> None:
    if process.state() == QProcess.ProcessState.NotRunning:
        return
    process_id = process.processId()
    isolated = os.name == "posix" and bool(process.property("isolated_process_group")) and process_id > 0
    if isolated:
        try:
            os.killpg(process_id, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError:
            process.terminate()
    elif os.name == "nt" and process_id > 0:
        # QProcess only owns the direct child on Windows; taskkill's tree mode also stops its workers.
        subprocess.run(["taskkill", "/PID", str(process_id), "/T"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    else:
        process.terminate()
    if wait_ms and not process.waitForFinished(wait_ms):
        if isolated:
            try:
                os.killpg(process_id, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        elif os.name == "nt" and process_id > 0:
            subprocess.run(["taskkill", "/PID", str(process_id), "/T", "/F"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
        else:
            process.kill()
        process.waitForFinished(1_000)


class JobRunner(QObject):
    """Owns the single running job and the history of jobs of this session."""

    job_started = Signal(object)
    job_output = Signal(object, str)
    job_finished = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.jobs: list[Job] = []
        self._process: QProcess | None = None
        self._job: Job | None = None

    @property
    def busy(self) -> bool:
        return self._process is not None

    @property
    def current(self) -> Job | None:
        return self._job

    def busy_reason(self) -> str:
        if self._job is None:
            return ""
        return (f"Another job is running ({self._job.label}). Only one job runs at a time; "
                "wait for it or terminate it in the console.")

    def start_nnunet(self, title: str, dataset: str, command: str) -> bool:
        """Run an nnU-Net command through ``nnworkbench-run`` so the saved paths are used."""
        from nnworkbench.services.commands import wrapped_command

        wrapped = wrapped_command(command)
        arguments = shlex.split(wrapped)
        return self._start(title, dataset, wrapped, arguments[0], arguments[1:])

    def start_analysis(self, dataset: str, raw_path: Path, workers: int) -> bool:
        arguments = ["-m", "nnworkbench.services.label_analysis", str(raw_path), "--workers", str(workers)]
        return self._start("Analyze dataset", dataset, f"label analysis of {raw_path.name}", sys.executable, arguments)

    def _start(self, title: str, dataset: str, display: str, program: str, arguments: list[str]) -> bool:
        if self.busy:
            return False
        job = Job(title, dataset, display)
        job.output.append(f"$ {display}\n")
        self.jobs.append(job)
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self._read(process, job))
        process.finished.connect(lambda code, status: self._finished(process, job, code, status))
        process.errorOccurred.connect(lambda error: self._failed(process, job, error))
        self._process, self._job = process, job
        print(f"Run: {display}")
        start_process_group(process, program, arguments)
        self.job_started.emit(job)
        self.busy_changed.emit(True)
        return True

    def terminate(self) -> None:
        if self._process is None or self._job is None:
            return
        self._job.status = "terminated"
        self._append(self._job, "\nTermination requested.\n")
        process = self._process
        stop_process_group(process)
        QTimer.singleShot(5_000, lambda: stop_process_group(process, wait_ms=1))

    def shutdown(self) -> None:
        if self._process is not None:
            stop_process_group(self._process, wait_ms=2_000)
        self._process = None

    def _read(self, process: QProcess, job: Job) -> None:
        text = bytes(process.readAllStandardOutput()).decode(errors="replace")
        if text:
            print(text, end="")
            self._append(job, text)

    def _append(self, job: Job, text: str) -> None:
        job.output.append(text)
        lines = [line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()]
        if lines:
            job.last_line = lines[-1]
        self.job_output.emit(job, text)

    def _finished(self, process: QProcess, job: Job, code: int, status) -> None:
        self._read(process, job)
        if job.status == "running":
            job.status = "finished" if status == QProcess.ExitStatus.NormalExit and code == 0 else "failed"
        self._append(job, f"\n[{job.status}, exit code {code}]\n")
        self._clear(process, job)

    def _failed(self, process: QProcess, job: Job, error) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return
        job.status = "failed"
        self._append(job, f"Could not start: {process.errorString()}\n")
        self._clear(process, job)

    def _clear(self, process: QProcess, job: Job) -> None:
        if process is self._process:
            self._process, self._job = None, None
            process.deleteLater()
            self.job_finished.emit(job)
            self.busy_changed.emit(False)
