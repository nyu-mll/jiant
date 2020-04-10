import optuna
import argparse


def delete_trails(full_task_name, input_module):
    storage = "sqlite:///example.db"
    if input_module != "default":
        study_name = f"{full_task_name}_{input_module}"
    else:
        study_name = full_task_name
        input_module = "roberta-large"
    optuna.delete_study(study_name=study_name, storage=storage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete Optuna trails")
    parser.add_argument("--full-task-name", type=str)
    parser.add_argument(
        "--input-module",
        type=str,
        default="default",
        choices=["default", "roberta-large", "albert-xxlarge-v2"],
    )

    args = parser.parse_args()
    delete_trails(args.full_task_name, args.input_module)
