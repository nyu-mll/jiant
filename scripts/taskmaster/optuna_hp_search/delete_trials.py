import optuna
import argparse


def delete_trails(study_name, input_module):
    storage = "sqlite:///example.db"
    if input_module != "default":
        stored_name = f"{study_name}_{input_module}"
    else:
        stored_name = study_name
    optuna.delete_study(study_name=stored_name, storage=storage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete Optuna trails")
    parser.add_argument("--study-name", type=str)
    parser.add_argument(
        "--input-module",
        type=str,
        default="default",
        choices=["default", "roberta-large", "albert-xxlarge-v2"],
    )

    args = parser.parse_args()
    delete_trails(args.study_name, args.input_module)
