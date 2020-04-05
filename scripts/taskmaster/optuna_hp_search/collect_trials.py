import optuna
import argparse
import os
import json
import pandas
import numpy

RESULT_DIR = "/scratch/hl3236/jiant_results"


def collect_trails(study_name, input_module):
    storage = "sqlite:///example.db"
    if input_module != "default":
        stored_name = f"{study_name}_{input_module}"
    else:
        stored_name = study_name
    print(stored_name)
    output = pandas.DataFrame(
        {
            "task": study_name,
            "n_samples": 0,
            "n_trails": 0,
            "batch_size": -1,
            "lr": -1.0,
            "max_epochs": -1,
            "median": numpy.nan,
        },
        index=[study_name],
    )

    try:
        study = optuna.load_study(study_name=stored_name, storage=storage)
    except Exception:
        return output
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
    df.to_csv(csv_file, index=False)

    df_grouped = df.groupby(["batch_size", "lr", "max_epochs"], as_index=False).agg(
        {"value": ["median", "mean", "min", "max", "count"]}
    )
    df_grouped.columns = ["batch_size", "lr", "max_epochs", "median", "mean", "min", "max", "count"]
    df_grouped = df_grouped.sort_values(["median", "count"], ascending=False)
    df_grouped.reset_index(drop=True, inplace=True)
    print(df_grouped)
    csv_file = os.path.join(RESULT_DIR, "optuna_csv", f"optuna_{study_name}_agg.csv")
    df_grouped.to_csv(csv_file, index=False)
    if len(df_grouped) > 0:
        output.n_samples[0] = df_grouped.count[0]
        output.n_trails[0] = sum(df_grouped.count)
        output.batch_size[0] = df_grouped.batch_size[0]
        output.lr[0] = df_grouped.lr[0]
        output.max_epochs[0] = df_grouped.max_epochs[0]
        output.median[0] = df_grouped.median[0]
    return output


def collect_all_trails(input_module):
    with open("task_metadata.json", "r") as f:
        task_metadata = json.loads(f.read())

    results = pandas.concat(
        [collect_trails(study_name, input_module) for study_name, task in task_metadata.items()]
    )
    csv_file = os.path.join(RESULT_DIR, "optuna_csv", f"optuna_{input_module}_integrated.csv")
    results.to_csv(csv_file, index=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Optuna trails")
    parser.add_argument("--study-name", type=str)
    parser.add_argument(
        "--input-module",
        type=str,
        default="default",
        choices=["default", "roberta-large", "albert-xxlarge-v2"],
    )

    args = parser.parse_args()
    if args.study_name == "ALL":
        collect_all_trails(args.input_module)
    else:
        collect_trails(args.study_name, args.input_module)
