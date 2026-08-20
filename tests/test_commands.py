from nnworkbench.services.commands import prediction_command, preprocessing_command, training_command


def test_nnunet_command_generation() -> None:
    assert preprocessing_command(1, "3d_fullres", "ExperimentPlanner") == (
        "nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres -pl ExperimentPlanner --verify_dataset_integrity")
    assert preprocessing_command(1, "2d", "ExperimentPlanner", "--np 8") == (
        "nnUNetv2_plan_and_preprocess -d 1 -c 2d -pl ExperimentPlanner --verify_dataset_integrity --np 8")
    assert training_command(1, "custom_plans.json", "2d", "nnUNetTrainer", "all") == (
        "nnUNetv2_train 1 2d all -tr nnUNetTrainer -p custom_plans")
    assert training_command(1, "nnUNetPlans.json", "3d_fullres", "nnUNetTrainer", "0", "--npz") == (
        "nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans --npz")
    assert prediction_command(1, "/input path", "/output path", "nnUNetPlans", "3d_fullres") == (
        "nnUNetv2_predict -i '/input path' -o '/output path' -d 1 -c 3d_fullres -p nnUNetPlans")
    assert prediction_command(1, "/input", "/output", "nnUNetPlans", "2d", "--save_probabilities") == (
        "nnUNetv2_predict -i /input -o /output -d 1 -c 2d -p nnUNetPlans --save_probabilities")
