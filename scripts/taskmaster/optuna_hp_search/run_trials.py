import optuna
import json
import subprocess
import os
import argparse

from shared_settings import batch_size_to_accumulation


RESULT_DIR = "/scratch/hl3236/jiant_results/"
FAILED_RUN_DEFAULT = None


def run_trials(
    full_task_name,
    gpu_available,
    n_trials,
    input_module,
    max_epochs_override,
    lr_override,
    batch_size_override,
):
    storage = "sqlite:///example.db"
    study_name = f"{full_task_name}_{input_module}"
    # temporary code to cope with an early code design
    if input_module == "roberta-large":
        study_name = full_task_name
    # temporary code to cope with an early code design
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    with open("scripts/taskmaster/optuna_hp_search/task_metadata.json", "r") as f:
        task_metadata = json.loads(f.read())

    def run_one_trial(trial):
        task = task_metadata[full_task_name]
        task_name = task["task_name"]
        exp_name = f"optuna_{study_name}"
        run_name = f"trial_{trial.number}"

        training_size = task["training_size"]
        if full_task_name.endswith("-5k"):
            target_train_data_fraction = 5000 / training_size
            training_size = 5000
        elif full_task_name.endswith("-20k"):
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

        if max_epochs_override < 0:
            max_epochs = trial.suggest_categorical("epochs", max_epochs_candidates)
        else:
            max_epochs = max_epochs_override
        if lr_override < 0:
            lr = trial.suggest_categorical("lr", lr_candidates)
        else:
            lr = lr_override
        if batch_size_override < 0:
            batch_size = trial.suggest_categorical("bs", batch_size_candidate)
        else:
            batch_size = batch_size_override
        batch_size_limit = task[f'{input_module.split("-")[0]}_batch_size_limit']
        real_batch_size, accumulation_steps = batch_size_to_accumulation(
            batch_size_limit, batch_size, gpu_available
        )
        val_interval = min(training_size // batch_size, 2500)

        overrides = []
        overrides.append(f"exp_name={exp_name}")
        overrides.append(f"run_name={run_name}")
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
    parser = argparse.ArgumentParser(description="Run Optuna trials")
    parser.add_argument("--study-name", type=str)
    parser.add_argument("--gpu-available", type=int)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument(
        "--input-module",
        type=str,
        default="default",
        choices=["default", "roberta-large", "albert-xxlarge-v2"],
    )
    parser.add_argument("--max-epochs", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=-1.0)
    parser.add_argument("--batch-size", type=int, default=-1)

    args = parser.parse_args()
    run_trials(
        args.study_name,
        args.gpu_available,
        args.n_trials,
        args.input_module,
        args.max_epochs,
        args.lr,
        args.batch_size,
    )
