import os

import jiant.scripts.download_data.utils as download_utils
import jiant.utils.python.io as py_io


def download_squad_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    if task_name == "squad_v1":
        train_file = "train-v1.1.json"
        dev_file = "dev-v1.1.json"
        version_2_with_negative = False
    elif task_name == "squad_v2":
        train_file = "train-v2.0.json"
        dev_file = "dev-v2.0.json"
        version_2_with_negative = True
    else:
        raise KeyError(task_name)

    os.makedirs(task_data_path, exist_ok=True)
    train_path = os.path.join(task_data_path, train_file)
    val_path = os.path.join(task_data_path, dev_file)
    download_utils.download_file(
        url=f"https://rajpurkar.github.io/SQuAD-explorer/dataset/{train_file}",
        file_path=train_path,
    )
    download_utils.download_file(
        url=f"https://rajpurkar.github.io/SQuAD-explorer/dataset/{dev_file}", file_path=val_path,
    )
    py_io.write_json(
        data={
            "task": "squad",
            "paths": {"train": train_path, "val": val_path},
            "version_2_with_negative": version_2_with_negative,
            "name": task_name,
        },
        path=task_config_path,
    )
