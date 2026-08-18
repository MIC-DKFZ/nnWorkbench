from nnworkbench.services.commands import prediction_command, preprocessing_command, training_command


def test_nnunet_command_generation() -> None:
    assert preprocessing_command(1, "3d_fullres", "ExperimentPlanner") == (
        "nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres -pl ExperimentPlanner")
    assert training_command(1, "custom_plans.json", "2d", "nnUNetTrainer", "all") == (
        "nnUNetv2_train 1 2d all -tr nnUNetTrainer -p custom_plans")
    assert prediction_command(1, "/input path", "/output path", "nnUNetPlans", "3d_fullres") == (
        "nnUNetv2_predict -i '/input path' -o '/output path' -d 1 -c 3d_fullres -p nnUNetPlans")
