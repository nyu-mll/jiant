import optuna
import sys


def delete_trails(study_name):
    storage = "sqlite:///example.db"
    optuna.delete_study(study_name=study_name, storage=storage)


if __name__ == "__main__":
    delete_trails(sys.argv[1])
