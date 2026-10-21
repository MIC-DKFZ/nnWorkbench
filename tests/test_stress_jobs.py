"""Robustness of the job runner, console, background tasks and settings (GUI stress test J1–J9)."""

import io
import sys
import time
from pathlib import Path

from qtpy.QtCore import QCoreApplication, QEvent, QSettings, QStandardPaths
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QWidget

from nnworkbench.ui import context as context_module
from nnworkbench.ui import jobs as jobs_module
from nnworkbench.ui.console import LINE_LIMIT, ConsolePanel
from nnworkbench.ui.jobs import OUTPUT_LIMIT, Job, JobRunner, _log_name
from nnworkbench.ui.tasks import TaskRunner
from nnworkbench.ui.widgets import NumberField


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


def _wait(application, done, seconds: float = 20) -> bool:
    end = time.time() + seconds
    while not done() and time.time() < end:
        application.processEvents()
        time.sleep(0.01)
    return done()


def _step(code: str) -> tuple[str, str, list[str]]:
    return "", sys.executable, ["-c", code]


class _BrokenOutput(io.TextIOBase):
    """The GUI's stdout when it is a pipe nobody reads any more."""

    def write(self, text: str) -> int:
        raise BrokenPipeError(32, "Broken pipe")


def test_the_delayed_second_stop_ignores_a_finished_job(tmp_path: Path, monkeypatch) -> None:  # J1
    application = _application()
    delayed = []

    class _Timer:
        @staticmethod
        def singleShot(_milliseconds, callback) -> None:
            delayed.append(callback)

    monkeypatch.setattr(jobs_module, "QTimer", _Timer)
    runner = JobRunner(application, log_folder=tmp_path)
    assert runner._start("Sleep", "Cells", "sleep", [_step("import time; time.sleep(30)")])
    runner.terminate()
    assert _wait(application, lambda: not runner.busy)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)  # the QProcess is deleted now
    assert runner.jobs[-1].status == "terminated" and len(delayed) == 1
    delayed[0]()  # 5 s later: must not touch the deleted QProcess


def test_a_broken_gui_stdout_does_not_stop_jobs(tmp_path: Path, monkeypatch) -> None:  # J2
    application = _application()
    monkeypatch.setattr(sys, "stdout", _BrokenOutput())
    runner = JobRunner(application, log_folder=tmp_path)
    assert runner._start("Echo", "Zellen", "echo", [_step("print('hello from the job')")])
    assert _wait(application, lambda: not runner.busy)
    job = runner.jobs[-1]
    assert job.status == "finished" and "hello from the job" in "".join(job.output)
    # an output that cannot encode the text (Windows: "nnworkbench-gui > log.txt" in the ANSI code page)
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="ascii"))
    assert runner._start("Échantillon", "細胞", "echo", [_step("print('Größe 細胞')")])
    assert _wait(application, lambda: not runner.busy)
    job = runner.jobs[-1]
    assert job.status == "finished" and "Größe 細胞" in "".join(job.output)


def test_napari_opens_with_a_broken_gui_stdout(tmp_path: Path, monkeypatch) -> None:  # J2
    application = _application()
    started = []
    monkeypatch.setattr(context_module, "QSettings",
                        lambda *_: QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(context_module, "start_process_group",
                        lambda _process, program, arguments: started.append(arguments))
    monkeypatch.setattr(sys, "stdout", _BrokenOutput())
    context = context_module.AppContext(application)
    context.inspect_case(tmp_path / "case_0000.nii.gz")
    assert len(started) == 1
    context.shutdown()


def test_long_and_cjk_titles_give_short_unique_log_names(tmp_path: Path) -> None:  # J3
    application = _application()
    first, second = Job("细胞核" * 100 + " A", "数据集" * 50, "x"), Job("细胞核" * 100 + " B", "数据集" * 50, "x")
    assert len(_log_name(first).encode("utf-8")) < 160 and _log_name(first) != _log_name(second)
    runner = JobRunner(application, log_folder=tmp_path)
    assert runner._start("细胞核" * 100, "数据集" * 50, "echo", [_step("print('done')")])
    assert _wait(application, lambda: not runner.busy)
    job = runner.jobs[-1]
    assert job.status == "finished" and "done" in job.log.read_text(encoding="utf-8")


def test_a_task_raising_system_exit_reports_a_failure() -> None:  # J4
    application = _application()
    tasks = TaskRunner(application)
    failures = []

    def task() -> None:
        raise SystemExit(2)

    tasks.run("exit", task, lambda _result: None, failures.append)
    assert _wait(application, lambda: failures, seconds=10)
    assert failures == ["SystemExit: 2"]
    tasks.shutdown()


def test_the_console_shows_a_job_running_in_the_background() -> None:  # J5
    application = _application()
    host = QWidget()
    runner = JobRunner(application)
    console = ConsolePanel(runner, host)
    runner.jobs.append(Job("Train · fold 0", "Cells", "x", status="running in the background"))
    console._refresh()
    console.toggle()  # expanded: the chips show the status
    host.close()


def test_output_without_line_breaks_is_broken_into_short_blocks() -> None:  # J6
    application = _application()
    host = QWidget()
    runner = JobRunner(application)
    console = ConsolePanel(runner, host)
    job = Job("Dots", "Cells", "x")
    runner.jobs.append(job)
    console._select(job)
    for _ in range(50):
        runner.job_output.emit(job, "." * 777)
    runner.job_output.emit(job, "\nnext line\n")
    document = console._log.document()
    lengths = [document.findBlockByNumber(number).length() - 1 for number in range(document.blockCount())]
    assert max(lengths) <= LINE_LIMIT and sum(lengths) >= 50 * 777
    assert console._log.toPlainText().endswith("\nnext line\n")
    host.close()


def test_the_output_kept_in_memory_is_bounded(tmp_path: Path) -> None:  # J7
    application = _application()
    host = QWidget()
    runner = JobRunner(application)
    console = ConsolePanel(runner, host)
    job = Job("Chatty", "Cells", "x", log=tmp_path / "chatty.log")
    runner.jobs.append(job)
    line = "x" * 99 + "\n"
    for number in range(60):
        runner._append(job, f"chunk {number}\n" + line * 1_000)  # 6 MB in all
    output = "".join(job.output)
    assert len(output) <= OUTPUT_LIMIT * 1.25 + 200 and job.truncated
    assert output.startswith("[Earlier output is not shown here") and str(job.log) in output.splitlines()[0]
    assert "chunk 59\n" in output and "chunk 0\n" not in output
    console._select(job)
    assert console._log.toPlainText().startswith("[Earlier output")
    host.close()


def test_saved_settings_are_clamped_to_the_dialog_limits(tmp_path: Path, monkeypatch) -> None:  # J8
    application = _application()
    path = str(tmp_path / "settings.ini")
    store = QSettings(path, QSettings.Format.IniFormat)
    store.setValue("analysis/workers", -5)
    store.setValue("napari/prefetch_radius", 999)
    store.sync()
    monkeypatch.setattr(context_module, "QSettings", lambda *_: QSettings(path, QSettings.Format.IniFormat))
    context = context_module.AppContext(application)
    assert context.settings.workers == 1 and context.settings.prefetch_radius == 20
    context.shutdown()
    store.setValue("analysis/workers", "many")
    store.sync()
    context = context_module.AppContext(application)
    assert context.settings.workers >= 1
    context.shutdown()


def test_a_number_with_a_plus_sign_is_taken() -> None:  # J9
    application = _application()
    host = QWidget()
    field = NumberField(1, 99, 3, host)
    field.selectAll()
    QTest.keyClicks(field, "+5")
    assert field.value() == 5
    field.editingFinished.emit()
    assert field.text() == "5"
    host.close()
    assert application is not None
