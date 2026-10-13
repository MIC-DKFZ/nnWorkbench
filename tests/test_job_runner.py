import sys
import time

from tests.test_ui_smoke import _application

from nnworkbench.ui.jobs import JobRunner


def _wait(application, runner: JobRunner, seconds: float = 20) -> None:
    end = time.time() + seconds
    while runner.busy and time.time() < end:
        application.processEvents()
        time.sleep(0.01)


def _step(label: str, code: str) -> tuple[str, str, list[str]]:
    return label, sys.executable, ["-c", code]


def test_steps_run_one_after_another() -> None:
    application = _application()
    runner = JobRunner(application)  # owned by the application, like in the GUI
    steps = [_step("fold 0", "print('Epoch 3')"), _step("fold 1", "print('Epoch 7')")]
    assert runner._start("Train · folds 0, 1", "Cells", "two folds", steps)
    _wait(application, runner)
    job = runner.jobs[-1]
    output = "".join(job.output)
    assert job.status == "finished" and output.count("$ ") == 2 and "[step finished" in output
    assert job.progress == "fold 1 (2/2) · epoch 7"


def test_steps_stop_at_the_first_failure() -> None:
    application = _application()
    runner = JobRunner(application)  # owned by the application, like in the GUI
    steps = [_step("fold 0", "import sys; sys.exit(3)"), _step("fold 1", "print('never')")]
    assert runner._start("Train · folds 0, 1", "Cells", "two folds", steps)
    _wait(application, runner)
    job = runner.jobs[-1]
    output = "".join(job.output)
    assert job.status == "failed" and job.exit_code == 3 and "never" not in output
    assert "not started: fold 1" in output
