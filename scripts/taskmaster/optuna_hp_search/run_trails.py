import optuna
import sys
import json

MAX_GPU = 4


def run_trails(study_name):
    storage = "sqlite:///example.db"
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize", load_if_exists=True
    )
    with open("task_metadata.json", "r") as f:
        task_metadata = json.loads(f.read())
    task = task_metadata[study_name]

    def run_one_trail(trail):
        task_name = task["task_name"]
        batch_size_limit = task["batch_size_limit"]
        training_size = task["training_size"]
        lr = [1e-5, 2e-5, 3e-5][trail.suggest_int("lr_select", 0, 2)]
        batch_size = [16, 32][trail.suggest_int("bs_select", 0, 1)]
        max_epochs_select = trail.suggest_int("epoch_select", 0, 1)
        if training_size <= 3000:
            max_epochs = [25, 50][max_epochs_select]
        elif training_size >= 300000:
            max_epochs = [2, 5][max_epochs_select]
        else:
            max_epochs = [7, 15][max_epochs_select]
        val_interval = min(training_size / batch_size, 2500)

        overrides = []
        overrides.append(f"lr={lr}")
        if batch_size // batch_size_limit <= MAX_GPU:
            overrides.append(f"batch_size={batch_size}")
            gpus = batch_size // batch_size_limit
        else:
            overrides.append()

    study.optimize(run_one_trail, sys.argv[2])


if __name__ == "__main__":
    run_trails(sys.argv[1])
