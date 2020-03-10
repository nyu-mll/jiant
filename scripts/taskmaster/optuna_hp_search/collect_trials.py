import optuna
import sys


def collect_trails(study_name):
    storage = "sqlite:///example.db"
    study = optuna.load_study(study_name=study_name, storage=storage)
    df = study.trials_dataframe()
    df = df[["number", "value", "user_attrs_batch_size", "user_attrs_lr", "user_attrs_max_epochs"]]
    df = df.rename(
        columns={
            "user_attrs_batch_size": "batch_size",
            "user_attrs_lr": "lr",
            "user_attrs_max_epochs": "max_epochs",
        }
    )
    df = df.dropna()
    df = df.sort_values(["batch_size", "lr", "max_epochs"])

    df_grouped = df.groupby(["batch_size", "lr", "max_epochs"], as_index=False).agg(
        {"value": ["median", "mean", "var", "count"]}
    )
    df_grouped.columns = ["batch_size", "lr", "max_epochs", "median", "mean", "var", "count"]
    df_grouped = df_grouped.sort_values(["median", "count"], ascending=False)
    print(df_grouped)
    csv_file = f"optuna_{study_name}.csv"
    df_grouped.to_csv(csv_file, index=True)


if __name__ == "__main__":
    collect_trails(sys.argv[1])
