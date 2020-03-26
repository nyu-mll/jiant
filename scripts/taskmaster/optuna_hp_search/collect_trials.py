import optuna
import sys
import os

RESULT_DIR = "/scratch/hl3236/jiant_results"


def collect_trails(study_name, input_module):
    storage = "sqlite:///example.db"
    if input_module != "None":
        stored_name = f"{study_name}_{input_module}"
    else:
        stored_name = study_name
    print(stored_name)
    study = optuna.load_study(study_name=stored_name, storage=storage)
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
    csv_file = os.path.join(RESULT_DIR, "optuna_csv", f"optuna_{study_name}_full.csv")
    df.to_csv(csv_file, index=True)

    df_grouped = df.groupby(["batch_size", "lr", "max_epochs"], as_index=False).agg(
        {"value": ["median", "mean", "min", "max", "count"]}
    )
    df_grouped.columns = ["batch_size", "lr", "max_epochs", "median", "mean", "min", "max", "count"]
    df_grouped = df_grouped.sort_values(["median", "count"], ascending=False)
    print(df_grouped)
    csv_file = os.path.join(RESULT_DIR, "optuna_csv", f"optuna_{study_name}_agg.csv")
    df_grouped.to_csv(csv_file, index=True)


if __name__ == "__main__":
    collect_trails(sys.argv[1], sys.argv[2])
