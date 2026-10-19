"""Run one command at a time (analysis, preprocessing, training, prediction) and keep its output."""

from __future__ import annotations

import os
import re
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
    progress: str = ""  # short hint for the console bar, e.g. "epoch 12" or "case 3/10"
    exit_code: int | None = None
    # A job can run several commands one after another (e.g. folds 0–4); it stops at the first failure.
    steps: list[tuple[str, str, list[str]]] = field(default_factory=list)  # (label, program, arguments) still to run
    step_prefix: str = ""  # e.g. "fold 2 (3/5) · " in front of the progress hint
    cases_total: int | None = None
    cases_done: int = 0

    @property
    def label(self) -> str:
        return f"{self.dataset} · {self.title}" if self.dataset else self.title


_EPOCH = re.compile(r"\bEpoch (\d+)\b")
_CASES_TO_PREDICT = re.compile(r"There are (\d+) cases that I would like to predict")
_PREDICTING = re.compile(r"^Predicting \S+:")
_COUNTER = re.compile(r"(\d+)/(\d+)")  # tqdm bars ("4/10 [00:03<…") and "Analyzed label 4/10: …"


def update_progress(job: Job, line: str) -> None:
    """Derive a short progress hint from one output line (training epochs, predicted cases, progress bars)."""
    if match := _EPOCH.search(line):
        job.progress = f"{job.step_prefix}epoch {match.group(1)}"  # the total is not in the output
    elif match := _CASES_TO_PREDICT.search(line):
        job.cases_total, job.cases_done = int(match.group(1)), 0
        job.progress = f"case 0/{job.cases_total}"
    elif _PREDICTING.match(line) and job.cases_total:
        job.cases_done = min(job.cases_done + 1, job.cases_total)
        job.progress = f"case {job.cases_done}/{job.cases_total}"
    elif ("|" in line or line.startswith("Analyzed label")) and (match := _COUNTER.search(line)):
        done, total = int(match.group(1)), int(match.group(2))
        if 0 < total and done <= total:
            job.progress = f"{done}/{total}"


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

    def start_nnunet(self, title: str, dataset: str, command: str | list[str], step_labels: list[str] | None = None
                     ) -> bool:
        """Run nnU-Net command(s) through ``nnworkbench-run`` so the saved paths are used. Several commands run
        one after another as one job (``step_labels`` names them in the progress hint, e.g. "fold 2")."""
        from nnworkbench.services.commands import wrapped_command

        commands = [command] if isinstance(command, str) else list(command)
        labels = step_labels or [""] * len(commands)
        steps = []
        for step_label, text in zip(labels, commands):
            # The same as "nnworkbench-run …", but started with this Python, so it works without the environment's
            # scripts folder on PATH (e.g. when the app is started from the macOS Dock or a Windows shortcut).
            arguments = ["-m", "nnworkbench.command_runner", *shlex.split(text)]
            steps.append((step_label, sys.executable, arguments))
        display = " && ".join(wrapped_command(text) for text in commands)
        return self._start(title, dataset, display, steps)

    def start_analysis(self, dataset: str, raw_path: Path, workers: int) -> bool:
        arguments = ["-m", "nnworkbench.services.label_analysis", str(raw_path), "--workers", str(workers)]
        return self._start("Compute statistics", dataset, f"statistics of {raw_path.name}",
                           [("", sys.executable, arguments)])

    def _start(self, title: str, dataset: str, display: str, steps: list[tuple[str, str, list[str]]]) -> bool:
        if self.busy or not steps:
            return False
        job = Job(title, dataset, display, steps=list(steps))
        self.jobs.append(job)
        self._launch(job, total=len(steps))
        self.job_started.emit(job)
        self.busy_changed.emit(True)
        return True

    def _launch(self, job: Job, total: int) -> None:
        """Start the next step of *job*."""
        step_label, program, arguments = job.steps.pop(0)
        number = total - len(job.steps)
        job.step_prefix = f"{step_label} ({number}/{total}) · " if total > 1 else ""
        job.progress = job.step_prefix.removesuffix(" · ")
        runner = ["-m", "nnworkbench.command_runner"]
        shown = (shlex.join(["nnworkbench-run", *arguments[2:]]) if arguments[:2] == runner
                 else shlex.join([program, *arguments]))
        if number == 1:
            job.output.append(f"$ {shown}\n")
        else:
            self._append(job, f"\n$ {shown}\n")
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")  # the output is read as UTF-8 (Windows default: ANSI code page)
        environment.insert("NNWORKBENCH_YES", "1")  # the GUI asks before replacing a training (command_runner)
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self._read(process, job))
        process.finished.connect(lambda code, status: self._finished(process, job, code, status, total))
        process.errorOccurred.connect(lambda error: self._failed(process, job, error))
        self._process, self._job = process, job
        print(f"Run: {shown}")
        start_process_group(process, program, arguments)

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
        for line in lines:
            update_progress(job, line)
        if lines:
            job.last_line = lines[-1]
        self.job_output.emit(job, text)

    def _finished(self, process: QProcess, job: Job, code: int, status, total: int = 1) -> None:
        self._read(process, job)
        job.exit_code = code
        succeeded = status == QProcess.ExitStatus.NormalExit and code == 0
        if job.status == "running" and succeeded and job.steps and process is self._process:
            self._append(job, f"\n[step finished, exit code {code}]\n")
            process.deleteLater()
            self._launch(job, total)  # next fold; the job stays the running one
            return
        if job.steps and not succeeded and job.status == "running":
            skipped = ", ".join(step[0] or step[1] for step in job.steps)
            self._append(job, f"\nStopped after this failure; not started: {skipped}\n")
        job.steps.clear()
        if job.status == "running":
            job.status = "finished" if status == QProcess.ExitStatus.NormalExit and code == 0 else "failed"
        output_line = job.last_line  # keep the last real output line (often the error) for the console bar
        self._append(job, f"\n[{job.status}, exit code {code}]\n")
        job.last_line = output_line
        self._clear(process, job)

    def _failed(self, process: QProcess, job: Job, error) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return
        job.status = "failed"
        job.steps.clear()
        self._append(job, f"Could not start: {process.errorString()}\n")
        self._clear(process, job)

    def _clear(self, process: QProcess, job: Job) -> None:
        if process is self._process:
            self._process, self._job = None, None
            process.deleteLater()
            self.job_finished.emit(job)
            self.busy_changed.emit(False)
