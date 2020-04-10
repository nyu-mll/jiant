import optuna
import argparse
import os
import json
import pandas
import numpy

RESULT_DIR = "/scratch/hl3236/jiant_results"


def collect_trails(full_task_name, input_module):
    storage = "sqlite:///example.db"
    if input_module != "default":
        study_name = f"{full_task_name}_{input_module}"
    else:
        study_name = full_task_name
        input_module = "roberta-large"
    print(study_name)
    output = pandas.DataFrame(
        {
            "task": full_task_name,
            "n_samples": 0,
            "n_trails": 0,
            "batch_size": -1,
            "lr": -1.0,
            "max_epochs": -1,
            "median": numpy.nan,
        },
        index=[full_task_name],
    )

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        return output
    df = study.trials_dataframe()
    try:
        df = df[
            ["number", "value", "user_attrs_batch_size", "user_attrs_lr", "user_attrs_max_epochs"]
        ]
    except Exception:
        return output
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
    df.to_csv(csv_file, index=False)
    if len(df) == 0:
        return output

    df_grouped = df.groupby(["batch_size", "lr", "max_epochs"], as_index=False).agg(
        {"value": ["median", "mean", "min", "max", "count"]}
    )
    df_grouped.columns = ["batch_size", "lr", "max_epochs", "median", "mean", "min", "max", "count"]
    df_grouped = df_grouped.sort_values(["median", "count"], ascending=False)
    df_grouped.reset_index(drop=True, inplace=True)
    print(df_grouped)
    csv_file = os.path.join(RESULT_DIR, "optuna_csv", f"optuna_{study_name}_agg.csv")
    df_grouped.to_csv(csv_file, index=False)
    output = pandas.DataFrame(
        {
            "task": full_task_name,
            "n_samples": df_grouped["count"][0],
            "n_trails": sum(df_grouped["count"]),
            "batch_size": df_grouped["batch_size"][0],
            "lr": df_grouped["lr"][0],
            "max_epochs": df_grouped["max_epochs"][0],
            "median": df_grouped["median"][0],
        },
        index=[full_task_name],
    )
    return output, df_grouped


def collect_all_trails(input_module):
    metadata_file = os.path.join(os.path.dirname(__file__), "task_metadata.json")
    with open(metadata_file, "r") as f:
        task_metadata = json.loads(f.read())

    results = pandas.concat(
        [
            collect_trails(full_task_name, input_module)[0]
            for full_task_name, task in task_metadata.items()
        ]
    )
    csv_file = os.path.join(RESULT_DIR, "optuna_csv", f"optuna_{input_module}_integrated.csv")
    results.to_csv(csv_file, index=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Optuna trails")
    parser.add_argument("--full-task-name", type=str)
    parser.add_argument(
        "--input-module",
        type=str,
        default="default",
        choices=["default", "roberta-large", "albert-xxlarge-v2"],
    )

    args = parser.parse_args()
    if args.full_task_name == "ALL":
        collect_all_trails(args.input_module)
    else:
        collect_trails(args.full_task_name, args.input_module)
