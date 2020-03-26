import optuna
import sys
import json
import subprocess
import os

RESULT_DIR = "/scratch/hl3236/jiant_results/"
FAILED_RUN_DEFAULT = None


def run_trials(study_name, gpu_available, n_trials, input_module):
    storage = "sqlite:///example.db"
    if input_module != "None":
        stored_name = f"{study_name}_{input_module}"
    else:
        stored_name = study_name
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name=stored_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    with open("scripts/taskmaster/optuna_hp_search/task_metadata.json", "r") as f:
        task_metadata = json.loads(f.read())

    def run_one_trial(trial):
        task = task_metadata[study_name]
        task_name = task["task_name"]
        exp_name = f"optuna_{study_name}_{input_module}"
        run_name = f"trial_{trial.number}"

        training_size = task["training_size"]
        if study_name.endswith("-5k"):
            target_train_data_fraction = 5000 / training_size
            training_size = 5000
        elif study_name.endswith("-20k"):
            target_train_data_fraction = 20000 / training_size
            training_size = 20000
        else:
            target_train_data_fraction = None

        if training_size <= 3000:
            max_epochs_candidates = [25, 50]
            batch_size_candidate = [8, 16]
            lr_candidates = [5e-6, 1e-5, 2e-5]
        elif training_size >= 300000:
            max_epochs_candidates = [2, 5]
            batch_size_candidate = [16, 32]
            lr_candidates = [1e-5, 2e-5, 3e-5]
        else:
            max_epochs_candidates = [7, 15]
            batch_size_candidate = [16, 32]
            lr_candidates = [1e-5, 2e-5, 3e-5]

        max_epochs = trial.suggest_categorical("epochs", max_epochs_candidates)
        lr = trial.suggest_categorical("lr", lr_candidates)
        batch_size = trial.suggest_categorical("bs", batch_size_candidate)
        batch_size_limit = task[f'{input_module.split("-")[0]}_batch_size_limit']
        gpu_needed = batch_size // batch_size_limit
        if gpu_needed <= gpu_available:
            real_batch_size = batch_size
            accumulation_steps = 1
        else:
            assert gpu_needed % gpu_available == 0
            accumulation_steps = gpu_needed // gpu_available
            assert batch_size % accumulation_steps == 0
            real_batch_size = batch_size // accumulation_steps
        val_interval = min(training_size // batch_size, 2500)

        overrides = []
        overrides.append(f"exp_name={exp_name}")
        overrides.append(f"run_name={run_name}")
        if input_module is not None:
            overrides.append(f"input_module={input_module}")
        overrides.append(f"pretrain_tasks=none")
        overrides.append(f"target_tasks={task_name}")
        overrides.append("do_target_task_training=1")
        overrides.append(f"max_epochs={max_epochs}")
        overrides.append(f"lr={lr}")
        overrides.append(f"batch_size={real_batch_size}")
        overrides.append(f"accumulation_steps={accumulation_steps}")
        overrides.append(f"target_train_val_interval={val_interval}")
        overrides.append("random_seed=-1")
        overrides.append("delete_checkpoints_when_done=1")
        if target_train_data_fraction is not None:
            overrides.append(f"target_train_data_fraction={target_train_data_fraction}")

        trial.set_user_attr("max_epochs", max_epochs)
        trial.set_user_attr("lr", lr)
        trial.set_user_attr("batch_size", batch_size)

        overrides = ", ".join(overrides)
        command = [
            "python",
            "main.py",
            "--config_file",
            "jiant/config/taskmaster/clean_roberta.conf",
            "--overrides",
            overrides,
        ]
        print(command)
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        process.wait()

        performance = FAILED_RUN_DEFAULT
        results_tsv = os.path.join(RESULT_DIR, exp_name, "results.tsv")
        if os.path.exists(results_tsv):
            with open(results_tsv, "r") as f:
                results = dict([line.split("\t") for line in f.read().split("\n") if line])
            if run_name in results:
                performance = float(results[run_name].split(", ")[0].split(": ")[-1])

        return performance

    study.optimize(run_one_trial, n_trials=n_trials)


if __name__ == "__main__":
    # python run_trials study_name gpu_available n_trials
    run_trials(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
