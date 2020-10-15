import logging
import os
import shutil

import jiant.utils.python.filesystem as filesystem
import jiant.scripts.download_data.utils as download_utils
import jiant.utils.python.io as py_io

from jiant.scripts.download_data.constants import SQUAD_TASKS, DIRECT_SUPERGLUE_TASKS_TO_DATA_URLS


def download_task_data_and_write_config(task_name: str, task_data_path: str, task_config_path: str):
    if task_name in SQUAD_TASKS:
        download_squad_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name in DIRECT_SUPERGLUE_TASKS_TO_DATA_URLS:
        download_superglue_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "abductive_nli":
        download_abductive_nli_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "fever_nli":
        download_fever_nli_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "swag":
        download_swag_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "qamr":
        download_qamr_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "qasrl":
        download_qasrl_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    else:
        raise KeyError(task_name)


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


def download_superglue_data_and_write_config(
        task_name: str, task_data_path: str, task_config_path: str
):
    # Applies to ReCoRD, MultiRC and WSC
    assert task_name not in SQUAD_TASKS

    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_unzip(
        DIRECT_SUPERGLUE_TASKS_TO_DATA_URLS[task_name], task_data_path
    )

    # Move task data up one folder (nested under task name when unzipped)
    # ie: mv ./record/ReCoRD/* ./record
    nested_task_dir = os.path.join(
        task_data_path, filesystem.find_case_insensitive_filename(task_name, task_data_path)
    )
    task_data_files = os.listdir(nested_task_dir)
    for f in task_data_files:
        # Overwrite file if it exists (overwrite by full path specification)
        shutil.move(os.path.join(nested_task_dir, f), os.path.join(task_data_path, f))
    shutil.rmtree(nested_task_dir)

    # Supports datasets with non-standard dev dataset name
    if os.path.isfile(os.path.join(task_data_path, "dev.jsonl")):
        dev_data_name = "dev.jsonl"
    elif os.path.isfile(os.path.join(task_data_path, "val.jsonl")):
        dev_data_name = "val.jsonl"
    else:
        raise RuntimeError("Unsupported dev dataset name in downloaded task.")

    val_path = os.path.join(task_data_path, dev_data_name)
    train_path = os.path.join(task_data_path, "train.jsonl")
    test_path = os.path.join(task_data_path, "test.jsonl")
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {"train": train_path, "val": val_path, "test": test_path},
            "name": task_name,
        },
        path=task_config_path,
    )


def download_abductive_nli_data_and_write_config(
        task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_unzip(
        "https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip",
        task_data_path,
    )
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train_inputs": os.path.join(task_data_path, "train.jsonl"),
                "train_labels": os.path.join(task_data_path, "train-labels.lst"),
                "val_inputs": os.path.join(task_data_path, "dev.jsonl"),
                "val_labels": os.path.join(task_data_path, "dev-labels.lst"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_fever_nli_data_and_write_config(task_name: str, task_data_path: str, task_config_path: str):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_unzip(
        ("https://www.dropbox.com/s/hylbuaovqwo2zav/nli_fever.zip?dl=1"),
        task_data_path,
    )
    # Since the FEVER NLI dataset doesn't have labels for the dev set, we also download the original
    # FEVER dev set and match example CIDs to obtain labels.
    orig_dev_path = os.path.join(task_data_path, "fever-dev-temp.jsonl")
    download_utils.download_file("https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl",
                                 orig_dev_path,
                                 )
    id_to_label = {}
    for line in py_io.read_jsonl(orig_dev_path):
        if "id" not in line:
            logging.warning("FEVER dev dataset is missing ID.")
            continue
        if "label" not in line:
            logging.warning("FEVER dev dataset is missing label.")
            continue
        id_to_label[line["id"]] = line["label"]
    os.remove(orig_dev_path)

    dev_path = os.path.join(task_data_path, "nli_fever", "dev_fitems.jsonl")
    dev_examples = []
    for line in py_io.read_jsonl(dev_path):
        if "cid" not in line:
            logging.warning("Data in {} is missing CID.".format(dev_path))
            continue
        if int(line["cid"]) not in id_to_label:
            logging.warning("Could not match CID {} to dev data.".format(line['cid']))
            continue
        dev_example = line
        dev_example["label"] = id_to_label[int(line["cid"])]
        dev_examples.append(dev_example)
    py_io.write_jsonl(dev_examples, os.path.join(task_data_path, "val.jsonl"))
    os.remove(dev_path)

    for phase in ["train", "test"]:
        os.rename(
            os.path.join(task_data_path, "nli_fever", f"{phase}_fitems.jsonl"),
            os.path.join(task_data_path, f"{phase}.jsonl"),
        )
    shutil.rmtree(os.path.join(task_data_path, "nli_fever"))

    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.jsonl"),
                "val": os.path.join(task_data_path, "val.jsonl"),
                "test": os.path.join(task_data_path, "test.jsonl"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_swag_data_and_write_config(task_name: str, task_data_path: str, task_config_path: str):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_unzip(
        "https://github.com/rowanz/swagaf/archive/master.zip", task_data_path,
    )
    for phase in ["train", "val", "test"]:
        os.rename(
            os.path.join(task_data_path, "swagaf-master", "data", f"{phase}.csv"),
            os.path.join(task_data_path, f"{phase}.csv"),
        )
    shutil.rmtree(os.path.join(task_data_path, "swagaf-master"))
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.csv"),
                "val": os.path.join(task_data_path, "val.csv"),
                "test": os.path.join(task_data_path, "test.csv"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_qamr_data_and_write_config(task_name: str, task_data_path: str, task_config_path: str):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_unzip(
        "https://github.com/uwnlp/qamr/archive/master.zip", task_data_path,
    )
    data_phase_list = ["train", "dev", "test"]
    jiant_phase_list = ["train", "val", "test"]
    for data_phase, jiant_phase in zip(data_phase_list, jiant_phase_list):
        os.rename(
            os.path.join(task_data_path, "qamr-master", "data", "filtered", f"{data_phase}.tsv"),
            os.path.join(task_data_path, f"{jiant_phase}.tsv"),
        )
    os.rename(
        os.path.join(task_data_path, "qamr-master", "data", "wiki-sentences.tsv"),
        os.path.join(task_data_path, "wiki-sentences.tsv"),
    )
    shutil.rmtree(os.path.join(task_data_path, "qamr-master"))
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.tsv"),
                "val": os.path.join(task_data_path, "val.tsv"),
                "test": os.path.join(task_data_path, "test.tsv"),
                "wiki_dict": os.path.join(task_data_path, "wiki-sentences.tsv"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_qasrl_data_and_write_config(
        task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_untar(
        "http://qasrl.org/data/qasrl-v2.tar", task_data_path,
    )
    data_phase_list = ["train", "dev", "test"]
    jiant_phase_list = ["train", "val", "test"]
    for data_phase, jiant_phase in zip(data_phase_list, jiant_phase_list):
        os.rename(
            os.path.join(task_data_path, "qasrl-v2", "orig", f"{data_phase}.jsonl.gz"),
            os.path.join(task_data_path, f"{jiant_phase}.jsonl.gz"),
        )
    shutil.rmtree(os.path.join(task_data_path, "qasrl-v2"))
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.jsonl.gz"),
                "val": os.path.join(task_data_path, "val.jsonl.gz"),
                "test": os.path.join(task_data_path, "test.jsonl.gz"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )
