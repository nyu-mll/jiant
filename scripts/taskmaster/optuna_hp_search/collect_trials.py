import optuna
import sys


def collect_trails(study_name):
    storage = "sqlite:///example.db"
    study = optuna.load_study(study_name=study_name, storage=storage)
    df = study.trials_dataframe()
    print(df)
    df_avg = df.groupby(["user_attrs_batch_size", "user_attrs_lr", "user_attrs_max_epochs"])[
        "value"
    ].mean()
    print(df_avg)
    csv_file = f"optuna_{study_name}.csv"
    df_avg.to_csv(csv_file, index=False)


if __name__ == "__main__":
    collect_trails(sys.argv[1])
