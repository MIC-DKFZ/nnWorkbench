from nnworkbench.ui.jobs import Job, update_progress


def _progress(command: str, lines: list[str]) -> str:
    job = Job("title", "dataset", command)
    for line in lines:
        update_progress(job, line)
    return job.progress


def test_training_epoch():
    lines = ["2026-10-13 10:00:00.1: Epoch 0", "2026-10-13 10:05:00.1: Epoch 12", "Pseudo dice [0.5]"]
    assert _progress("nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer_250epochs", lines) == "epoch 12"


def test_prediction_counts_cases():
    lines = ["There are 10 cases in the source folder", "There are 4 cases that I would like to predict",
             "Predicting a.nii.gz:", "Predicting b.nii.gz:"]
    assert _progress("nnUNetv2_predict", lines) == "case 2/4"


def test_progress_bars_and_statistics():
    bar = "Extracting dataset fingerprint:  40%|████      | 4/10 [00:03<00:05,  1.2it/s]"
    assert _progress("nnUNetv2_plan_and_preprocess", [bar]) == "4/10"
    assert _progress("statistics", ["Analyzed label 7/12: a.nii.gz"]) == "7/12"
    assert _progress("statistics", ["nothing to count here 2026/10"]) == ""
