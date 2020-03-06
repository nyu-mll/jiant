import optuna
import sys
import json
import subprocess
import os

RESULT_DIR = "/scratch/hl3236/jiant_results/"
FAILED_RUN_DEFAULT = -1.0


def run_trials(study_name, gpu_available, n_trials):
    storage = "sqlite:///example.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize", load_if_exists=True
    )
    with open("scripts/taskmaster/optuna_hp_search/task_metadata.json", "r") as f:
        task_metadata = json.loads(f.read())
    task = task_metadata[study_name]

    def run_one_trial(trial):
        task_name = task["task_name"]
        exp_name = f"optuna_{task_name}"
        run_name = f"trial_{trial.number}"

        training_size = task["training_size"]
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

        max_epochs_select = trial.suggest_int("epoch_select", 0, len(max_epochs_candidates) - 1)
        max_epochs = max_epochs_candidates[max_epochs_select]
        lr_select = trial.suggest_int("lr_select", 0, len(lr_candidates) - 1)
        lr = lr_candidates[lr_select]
        batch_size_select = trial.suggest_int("bs_select", 0, len(batch_size_candidate) - 1)
        batch_size = batch_size_candidate[batch_size_select]
        batch_size_limit = task["batch_size_limit"]
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
        overrides.append(f"pretrain_tasks=none")
        overrides.append(f"target_tasks={task_name}")
        overrides.append("do_target_task_training=1")
        overrides.append(f"max_epochs={max_epochs}")
        overrides.append(f"lr={lr}")
        overrides.append(f"batch_size={real_batch_size}")
        overrides.append(f"accumulation_steps={accumulation_steps}")
        overrides.append(f"target_train_val_interval={val_interval}")
        overrides.append("random_seed=-1")

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

        results_tsv = os.path.join(RESULT_DIR, exp_name, "results.tsv")
        with open(results_tsv, "r") as f:
            results = dict([line.split("\t") for line in f.read().split("\n") if line])
        if run_name in results:
            performance = float(results[run_name].split(", ")[0].split(": ")[-1])
        else:
            performance = FAILED_RUN_DEFAULT

        return performance

    study.optimize(run_one_trial, n_trials=n_trials)


if __name__ == "__main__":
    # python run_trials task_name gpu_available n_trials
    run_trials(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
