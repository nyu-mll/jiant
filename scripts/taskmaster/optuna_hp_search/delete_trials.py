import optuna
import sys


def delete_trails(study_name, input_module):
    storage = "sqlite:///example.db"
    if input_module != "None":
        stored_name = f"{study_name}_{input_module}"
    else:
        stored_name = study_name
    optuna.delete_study(study_name=stored_name, storage=storage)


if __name__ == "__main__":
    delete_trails(sys.argv[1], sys.argv[2])
