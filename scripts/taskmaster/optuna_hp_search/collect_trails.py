import optuna
import sys


def collect_trails(study_name):
    storage = "sqlite:///example.db"
    study = optuna.load_study(study_name=study_name, storage=storage)
    df = study.trials_dataframe()
    csv_file = f"{study_name}.csv"
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    collect_trails(sys.argv[1])
