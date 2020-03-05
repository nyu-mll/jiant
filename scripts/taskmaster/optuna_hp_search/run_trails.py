import optuna
import sys
import json
import subprocess
import os

RESULT_DIR = "/scratch/hl3236/jiant_results/"
FAILED_RUN_DEFAULT = -1.0


def run_trails(study_name, gpu_available, n_trails):
    storage = "sqlite:///example.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize", load_if_exists=True
    )
    with open("task_metadata.json", "r") as f:
        task_metadata = json.loads(f.read())
    task = task_metadata[study_name]

    def run_one_trail(trail):
        task_name = task["task_name"]
        exp_name = f"optuna_{task_name}"
        run_name = f"trail_{trail.num}"

        batch_size = [16, 32][trail.suggest_int("bs_select", 0, 1)]
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

        lr = [1e-5, 2e-5, 3e-5][trail.suggest_int("lr_select", 0, 2)]

        training_size = task["training_size"]
        max_epochs_select = trail.suggest_int("epoch_select", 0, 1)
        if training_size <= 3000:
            max_epochs = [25, 50][max_epochs_select]
        elif training_size >= 300000:
            max_epochs = [2, 5][max_epochs_select]
        else:
            max_epochs = [7, 15][max_epochs_select]
        val_interval = min(training_size / batch_size, 2500)

        overrides = []
        overrides.append(f"exp_name={exp_name}")
        overrides.append(f"run_name={run_name}")
        overrides.append(f"batch_size={real_batch_size}")
        overrides.append(f"accumulation_steps={accumulation_steps}")
        overrides.append(f"lr={lr}")
        overrides.append(f"max_epochs={max_epochs}")
        overrides.append(f"target_train_val_interval={val_interval}")

        trail.set_user_attr("batch_size", batch_size)
        trail.set_user_attr("lr", lr)
        trail.set_user_attr("max_epochs", max_epochs)

        command = [
            "python",
            "../../main.py",
            "--config_file",
            "jiant/config/taskmaster/clean_roberta.conf",
            "--overrides",
            f'"{", ".join(overrides)}"',
        ]
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

    study.optimize(run_one_trail, n_trails=n_trails)


if __name__ == "__main__":
    # python run_trails task_name gpu_available n_trails
    run_trails(sys.argv[1], sys.argv[2], sys.argv[3])
