import sys
import time
from pathlib import Path

from qtpy.QtCore import QStandardPaths
from qtpy.QtWidgets import QApplication

from nnworkbench.ui.jobs import JobRunner


def _application() -> QApplication:
    QStandardPaths.setTestModeEnabled(True)
    return QApplication.instance() or QApplication([])


def _wait(application, runner: JobRunner, seconds: float = 20) -> None:
    end = time.time() + seconds
    while runner.busy and time.time() < end:
        application.processEvents()
        time.sleep(0.01)


def _step(label: str, code: str) -> tuple[str, str, list[str]]:
    return label, sys.executable, ["-c", code]


def test_steps_run_one_after_another(tmp_path: Path) -> None:
    application = _application()
    runner = JobRunner(application, log_folder=tmp_path)  # owned by the application, like in the GUI
    steps = [_step("fold 0", "print('Epoch 3')"), _step("fold 1", "print('Epoch 7')")]
    assert runner._start("Train · folds 0, 1", "Cells", "two folds", steps)
    _wait(application, runner)
    job = runner.jobs[-1]
    output = "".join(job.output)
    assert job.status == "finished" and output.count("$ ") == 2 and "[step finished" in output
    assert job.progress == "fold 1 (2/2) · epoch 7"


def test_steps_stop_at_the_first_failure(tmp_path: Path) -> None:
    application = _application()
    runner = JobRunner(application, log_folder=tmp_path)  # owned by the application, like in the GUI
    steps = [_step("fold 0", "import sys; sys.exit(3)"), _step("fold 1", "print('never')")]
    assert runner._start("Train · folds 0, 1", "Cells", "two folds", steps)
    _wait(application, runner)
    job = runner.jobs[-1]
    output = "".join(job.output)
    assert job.status == "failed" and job.exit_code == 3 and "never" not in output
    assert "not started: fold 1" in output


def test_the_output_is_also_in_a_log_file(tmp_path: Path) -> None:
    application = _application()
    runner = JobRunner(application, log_folder=tmp_path)
    steps = [_step("fold 0", "print('Epoch 3')"), _step("fold 1", "print('Epoch 7')")]
    assert runner._start("Train · folds 0, 1", "Cells", "two folds", steps)
    _wait(application, runner)
    job = runner.jobs[-1]
    log = job.log.read_text(encoding="utf-8")
    assert job.log.parent == tmp_path and job.log.name.endswith("_Cells_Train_folds_0_1.log")
    assert log.count("$ ") == 2 and "Epoch 3" in log and "Epoch 7" in log
    assert f"Log file: {job.log}" in "".join(job.output)


def test_a_detached_job_keeps_running_and_writing_its_log(tmp_path: Path) -> None:
    application = _application()
    runner = JobRunner(application, log_folder=tmp_path)
    marker = tmp_path / "done.txt"
    # 'run' + 'ning': the command line in the output must not match before the command printed it
    code = (f"import time, pathlib; print('run' + 'ning', flush=True); time.sleep(1.5); print('after the GUI quit', "
            f"flush=True); pathlib.Path({str(marker)!r}).write_text('ok')")
    assert runner._start("Train · fold 0", "Cells", "one fold", [_step("fold 0", code)])
    end = time.time() + 10
    # print() ends lines with "\r\n" on Windows
    while "running\n" not in "".join(runner.current.output).replace("\r\n", "\n") and time.time() < end:
        application.processEvents()
        time.sleep(0.01)
    job = runner.detach()
    assert job is not None and not runner.busy and job.status == "running in the background"
    end = time.time() + 15
    while not marker.exists() and time.time() < end:
        time.sleep(0.05)
    assert marker.read_text() == "ok" and "after the GUI quit" in job.log.read_text(encoding="utf-8")


def test_the_runner_returns_the_exit_code_and_survives_a_closed_output(tmp_path: Path) -> None:
    import subprocess

    from nnworkbench.services.process_runner import run_logged

    log = tmp_path / "jobs" / "job.log"
    assert run_logged(sys.executable, ["-c", "print('x'); raise SystemExit(4)"], log, poll_seconds=0.01) == 4
    assert log.read_text(encoding="utf-8").strip() == "x"
    # the reader of the runner's output goes away: the command still finishes and its log is complete
    marker = tmp_path / "done.txt"
    code = f"import time, pathlib; time.sleep(0.5); print('late'); pathlib.Path({str(marker)!r}).write_text('ok')"
    runner = subprocess.Popen([sys.executable, "-m", "nnworkbench.services.process_runner", "--log", str(log),
                               sys.executable, "-c", code], stdout=subprocess.PIPE)
    runner.stdout.close()
    assert runner.wait(timeout=20) == 0 and marker.exists() and "late" in log.read_text(encoding="utf-8")
