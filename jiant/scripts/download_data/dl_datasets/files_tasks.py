import json
import logging
import os
import pandas as pd
import re
import shutil
import tarfile
from operator import itemgetter
from collections import Counter

import jiant.scripts.download_data.utils as download_utils
import jiant.utils.display as display
import jiant.utils.python.filesystem as filesystem
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
    elif task_name == "arct":
        download_arct_data_and_write_config(
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
    elif task_name == "newsqa":
        download_newsqa_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "mctaco":
        download_mctaco_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "mctest160":
        download_mctest160_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "mctest500":
        download_mctest500_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "mrqa_natural_questions":
        download_mrqa_natural_questions_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "mutual":
        download_mutual_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "mutual_plus":
        download_mutual_plus_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "piqa":
        download_piqa_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "winogrande":
        download_winogrande_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name == "ropes":
        download_ropes_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name in [
        "acceptability_definiteness",
        "acceptability_coord",
        "acceptability_eos",
        "acceptability_whwords",
    ]:
        download_acceptability_judgments_data_and_write_config(
            task_name=task_name, task_data_path=task_data_path, task_config_path=task_config_path
        )
    elif task_name in [
        "senteval_bigram_shift",
        "senteval_coordination_inversion",
        "senteval_obj_number",
        "senteval_odd_man_out",
        "senteval_past_present",
        "senteval_sentence_length",
        "senteval_subj_number",
        "senteval_top_constituents",
        "senteval_tree_depth",
        "senteval_word_content",
    ]:
        download_senteval_data_and_write_config(
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
            "kwargs": {"version_2_with_negative": version_2_with_negative},
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


def download_arct_data_and_write_config(task_name: str, task_data_path: str, task_config_path: str):
    os.makedirs(task_data_path, exist_ok=True)
    file_name_list = [
        "train-doubled.tsv",
        "train-w-swap-doubled.tsv",
        "train-w-swap.tsv",
        "train.tsv",
        "dev.tsv",
        "test.tsv",
    ]
    for file_name in file_name_list:
        download_utils.download_file(
            f"https://raw.githubusercontent.com/UKPLab/argument-reasoning-comprehension-task/"
            + f"master/experiments/src/main/python/data/{file_name}",
            os.path.join(task_data_path, file_name),
        )
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.tsv"),
                "val": os.path.join(task_data_path, "val.tsv"),
                "test": os.path.join(task_data_path, "test.tsv"),
                "train_doubled": os.path.join(task_data_path, "train-doubled.tsv"),
                "train_w_swap": os.path.join(task_data_path, "train-w-swap.tsv"),
                "train_w_swap_doubled": os.path.join(task_data_path, "train-w-swap-doubled.tsv"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_mctaco_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    file_name_list = ["dev_3783.tsv", "test_9442.tsv"]
    for file_name in file_name_list:
        download_utils.download_file(
            f"https://raw.githubusercontent.com/CogComp/MCTACO/master/dataset/{file_name}",
            os.path.join(task_data_path, file_name),
        )
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "val": os.path.join(task_data_path, "dev_3783.tsv"),
                "test": os.path.join(task_data_path, "test_9442.tsv"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_mctest160_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_unzip(
        "https://mattr1.github.io/mctest/data/MCTest.zip", task_data_path,
    )
    download_utils.download_and_unzip(
        "https://mattr1.github.io/mctest/data/MCTestAnswers.zip", task_data_path,
    )
    os.rename(
        os.path.join(task_data_path, "MCTestAnswers", f"mc160.test.ans"),
        os.path.join(task_data_path, "MCTest", f"mc160.test.ans"),
    )
    shutil.rmtree(os.path.join(task_data_path, "MCTestAnswers"))
    for phase in ["train", "dev", "test"]:
        os.rename(
            os.path.join(task_data_path, "MCTest", f"mc160.{phase}.tsv"),
            os.path.join(task_data_path, f"mc160.{phase}.tsv"),
        )
        os.rename(
            os.path.join(task_data_path, "MCTest", f"mc160.{phase}.ans"),
            os.path.join(task_data_path, f"mc160.{phase}.ans"),
        )
    shutil.rmtree(os.path.join(task_data_path, "MCTest"))

    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "mc160.train.tsv"),
                "train_ans": os.path.join(task_data_path, "mc160.train.ans"),
                "val": os.path.join(task_data_path, "mc160.dev.tsv"),
                "val_ans": os.path.join(task_data_path, "mc160.dev.ans"),
                "test": os.path.join(task_data_path, "mc160.test.tsv"),
                "test_ans": os.path.join(task_data_path, "mc160.test.ans"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_mctest500_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_unzip(
        "https://mattr1.github.io/mctest/data/MCTest.zip", task_data_path,
    )
    download_utils.download_and_unzip(
        "https://mattr1.github.io/mctest/data/MCTestAnswers.zip", task_data_path,
    )
    os.rename(
        os.path.join(task_data_path, "MCTestAnswers", f"mc500.test.ans"),
        os.path.join(task_data_path, "MCTest", f"mc500.test.ans"),
    )
    shutil.rmtree(os.path.join(task_data_path, "MCTestAnswers"))
    for phase in ["train", "dev", "test"]:
        os.rename(
            os.path.join(task_data_path, "MCTest", f"mc500.{phase}.tsv"),
            os.path.join(task_data_path, f"mc500.{phase}.tsv"),
        )
        os.rename(
            os.path.join(task_data_path, "MCTest", f"mc500.{phase}.ans"),
            os.path.join(task_data_path, f"mc500.{phase}.ans"),
        )
    shutil.rmtree(os.path.join(task_data_path, "MCTest"))

    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "mc500.train.tsv"),
                "train_ans": os.path.join(task_data_path, "mc500.train.ans"),
                "val": os.path.join(task_data_path, "mc500.dev.tsv"),
                "val_ans": os.path.join(task_data_path, "mc500.dev.ans"),
                "test": os.path.join(task_data_path, "mc500.test.tsv"),
                "test_ans": os.path.join(task_data_path, "mc500.test.ans"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_mutual_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    os.makedirs(task_data_path + "/train", exist_ok=True)
    os.makedirs(task_data_path + "/dev", exist_ok=True)
    os.makedirs(task_data_path + "/test", exist_ok=True)
    num_files = {"train": 7088, "dev": 886, "test": 886}
    for phase in num_files:
        examples = []
        for i in range(num_files[phase]):
            file_name = phase + "_" + str(i + 1) + ".txt"
            download_utils.download_file(
                f"https://raw.githubusercontent.com/Nealcly/MuTual/"
                + f"master/data/mutual/{phase}/{file_name}",
                os.path.join(task_data_path, phase, file_name),
            )
            for line in py_io.read_file_lines(os.path.join(task_data_path, phase, file_name)):
                examples.append(line)
        py_io.write_jsonl(examples, os.path.join(task_data_path, phase + ".jsonl"))
        shutil.rmtree(os.path.join(task_data_path, phase))

    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.jsonl"),
                "val": os.path.join(task_data_path, "dev.jsonl"),
                "test": os.path.join(task_data_path, "test.jsonl"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_mutual_plus_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    os.makedirs(task_data_path + "/train", exist_ok=True)
    os.makedirs(task_data_path + "/dev", exist_ok=True)
    os.makedirs(task_data_path + "/test", exist_ok=True)
    num_files = {"train": 7088, "dev": 886, "test": 886}
    for phase in num_files:
        examples = []
        for i in range(num_files[phase]):
            file_name = phase + "_" + str(i + 1) + ".txt"
            download_utils.download_file(
                f"https://raw.githubusercontent.com/Nealcly/MuTual/"
                + f"master/data/mutual_plus/{phase}/{file_name}",
                os.path.join(task_data_path, phase, file_name),
            )
            for line in py_io.read_file_lines(os.path.join(task_data_path, phase, file_name)):
                examples.append(line)
        py_io.write_jsonl(examples, os.path.join(task_data_path, phase + ".jsonl"))
        shutil.rmtree(os.path.join(task_data_path, phase))

    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.jsonl"),
                "val": os.path.join(task_data_path, "dev.jsonl"),
                "test": os.path.join(task_data_path, "test.jsonl"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_fever_nli_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_unzip(
        "https://www.dropbox.com/s/hylbuaovqwo2zav/nli_fever.zip?dl=1", task_data_path,
    )
    # Since the FEVER NLI dataset doesn't have labels for the dev set, we also download the original
    # FEVER dev set and match example CIDs to obtain labels.
    orig_dev_path = os.path.join(task_data_path, "fever-dev-temp.jsonl")
    download_utils.download_file(
        "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl", orig_dev_path,
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
            logging.warning("Could not match CID {} to dev data.".format(line["cid"]))
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


def download_newsqa_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    def get_consensus_answer(row_):
        answer_char_start, answer_char_end = None, None
        if row_.validated_answers:
            validated_answers_ = json.loads(row.validated_answers)
            answer_, max_count = max(validated_answers_.items(), key=itemgetter(1))
            total_count = sum(validated_answers_.values())
            if max_count >= total_count / 2.0:
                if answer_ != "none" and answer_ != "bad_question":
                    answer_char_start, answer_char_end = map(int, answer_.split(":"))
                else:
                    # No valid answer.
                    pass
        else:
            # Check row_.answer_char_ranges for most common answer.
            # No validation was done so there must be an answer with consensus.
            answers = Counter()
            for user_answer in row_.answer_char_ranges.split("|"):
                for ans in user_answer.split(","):
                    answers[ans] += 1
            top_answer = answers.most_common(1)
            if top_answer:
                top_answer, _ = top_answer[0]
                if ":" in top_answer:
                    answer_char_start, answer_char_end = map(int, top_answer.split(":"))

        return answer_char_start, answer_char_end

    def load_combined(path):
        result = pd.read_csv(
            path,
            encoding="utf-8",
            dtype=dict(is_answer_absent=float),
            na_values=dict(question=[], story_text=[], validated_answers=[]),
            keep_default_na=False,
        )

        if "story_text" in result.keys():
            for row_ in display.tqdm(
                result.itertuples(), total=len(result), desc="Adjusting story texts"
            ):
                story_text_ = row_.story_text.replace("\r\n", "\n")
                result.at[row_.Index, "story_text"] = story_text_

        return result

    def _map_answers(answers):
        result = []
        for a in answers.split("|"):
            user_answers = []
            result.append(dict(sourcerAnswers=user_answers))
            for r in a.split(","):
                if r == "None":
                    user_answers.append(dict(noAnswer=True))
                else:
                    start_, end_ = map(int, r.split(":"))
                    user_answers.append(dict(s=start_, e=end_))
        return result

    def strip_empty_strings(strings):
        while strings and strings[-1] == "":
            del strings[-1]
        return strings

    # Require: cnn_stories.tgz
    cnn_stories_path = os.path.join(task_data_path, "cnn_stories.tgz")
    assert os.path.exists(cnn_stories_path), (
        "Download CNN Stories from https://cs.nyu.edu/~kcho/DMQA/ and save to " + cnn_stories_path
    )
    # Require: newsqa-data-v1/newsqa-data-v1.csv
    dataset_path = os.path.join(task_data_path, "newsqa-data-v1", "newsqa-data-v1.csv")
    if os.path.exists(dataset_path):
        pass
    elif os.path.exists(os.path.join(task_data_path, "newsqa-data-v1.zip")):
        download_utils.unzip_file(
            zip_path=os.path.join(task_data_path, "newsqa-data-v1.zip"),
            extract_location=task_data_path,
            delete=False,
        )
    else:
        raise AssertionError(
            "Download https://www.microsoft.com/en-us/research/project/newsqa-dataset/#!download"
            " and save to " + os.path.join(task_data_path, "newsqa-data-v1.zip")
        )

    # Download auxiliary data
    os.makedirs(task_data_path, exist_ok=True)
    file_name_list = [
        "train_story_ids.csv",
        "dev_story_ids.csv",
        "test_story_ids.csv",
        "stories_requiring_extra_newline.csv",
        "stories_requiring_two_extra_newlines.csv",
        "stories_to_decode_specially.csv",
    ]
    for file_name in file_name_list:
        download_utils.download_file(
            f"https://raw.githubusercontent.com/Maluuba/newsqa/master/maluuba/newsqa/{file_name}",
            os.path.join(task_data_path, file_name),
        )

    dataset = load_combined(dataset_path)
    remaining_story_ids = set(dataset["story_id"])
    with open(
        os.path.join(task_data_path, "stories_requiring_extra_newline.csv"), "r", encoding="utf-8"
    ) as f:
        stories_requiring_extra_newline = set(f.read().split("\n"))

    with open(
        os.path.join(task_data_path, "stories_requiring_two_extra_newlines.csv"),
        "r",
        encoding="utf-8",
    ) as f:
        stories_requiring_two_extra_newlines = set(f.read().split("\n"))

    with open(
        os.path.join(task_data_path, "stories_to_decode_specially.csv"), "r", encoding="utf-8"
    ) as f:
        stories_to_decode_specially = set(f.read().split("\n"))

    # Start combining data files
    story_id_to_text = {}
    with tarfile.open(cnn_stories_path, mode="r:gz", encoding="utf-8") as t:
        highlight_indicator = "@highlight"

        copyright_line_pattern = re.compile(
            "^(Copyright|Entire contents of this article copyright, )"
        )
        with display.tqdm(total=len(remaining_story_ids), desc="Getting story texts") as pbar:
            for member in t.getmembers():
                story_id = member.name
                if story_id in remaining_story_ids:
                    remaining_story_ids.remove(story_id)
                    story_file = t.extractfile(member)

                    # Correct discrepancies in stories.
                    # Problems are caused by using several programming languages and libraries.
                    # When ingesting the stories, we started with Python 2.
                    # After dealing with unicode issues, we tried switching to Python 3.
                    # That caused inconsistency problems so we switched back to Python 2.
                    # Furthermore, when crowdsourcing, JavaScript and HTML templating perturbed
                    # the stories.
                    # So here we map the text to be compatible with the indices.
                    lines = map(lambda s_: s_.strip().decode("utf-8"), story_file.readlines())

                    story_file.close()
                    lines = list(lines)
                    highlights_start = lines.index(highlight_indicator)
                    story_lines = lines[:highlights_start]
                    story_lines = strip_empty_strings(story_lines)
                    while len(story_lines) > 1 and copyright_line_pattern.search(story_lines[-1]):
                        story_lines = strip_empty_strings(story_lines[:-2])
                    if story_id in stories_requiring_two_extra_newlines:
                        story_text = "\n\n\n".join(story_lines)
                    elif story_id in stories_requiring_extra_newline:
                        story_text = "\n\n".join(story_lines)
                    else:
                        story_text = "\n".join(story_lines)

                    story_text = story_text.replace("\xe2\x80\xa2", "\xe2\u20ac\xa2")
                    story_text = story_text.replace("\xe2\x82\xac", "\xe2\u201a\xac")
                    story_text = story_text.replace("\r", "\n")
                    if story_id in stories_to_decode_specially:
                        story_text = story_text.replace("\xe9", "\xc3\xa9")
                    story_id_to_text[story_id] = story_text

                    pbar.update()

                    if len(remaining_story_ids) == 0:
                        break

    for row in display.tqdm(dataset.itertuples(), total=len(dataset), desc="Setting story texts"):
        # Set story_text since we cannot include it in the dataset.
        story_text = story_id_to_text[row.story_id]
        dataset.at[row.Index, "story_text"] = story_text

        # Handle endings that are too large.
        answer_char_ranges = row.answer_char_ranges.split("|")
        updated_answer_char_ranges = []
        ranges_updated = False
        for user_answer_char_ranges in answer_char_ranges:
            updated_user_answer_char_ranges = []
            for char_range in user_answer_char_ranges.split(","):
                if char_range != "None":
                    start, end = map(int, char_range.split(":"))
                    if end > len(story_text):
                        ranges_updated = True
                        end = len(story_text)
                    if start < end:
                        updated_user_answer_char_ranges.append("%d:%d" % (start, end))
                    else:
                        # It's unclear why but sometimes the end is after the start.
                        # We'll filter these out.
                        ranges_updated = True
                else:
                    updated_user_answer_char_ranges.append(char_range)
            if updated_user_answer_char_ranges:
                updated_user_answer_char_ranges = ",".join(updated_user_answer_char_ranges)
                updated_answer_char_ranges.append(updated_user_answer_char_ranges)
        if ranges_updated:
            updated_answer_char_ranges = "|".join(updated_answer_char_ranges)
            dataset.at[row.Index, "answer_char_ranges"] = updated_answer_char_ranges

        if row.validated_answers and not pd.isnull(row.validated_answers):
            updated_validated_answers = {}
            validated_answers = json.loads(row.validated_answers)
            for char_range, count in validated_answers.items():
                if ":" in char_range:
                    start, end = map(int, char_range.split(":"))
                    if end > len(story_text):
                        ranges_updated = True
                        end = len(story_text)
                    if start < end:
                        char_range = "{}:{}".format(start, end)
                        updated_validated_answers[char_range] = count
                    else:
                        # It's unclear why but sometimes the end is after the start.
                        # We'll filter these out.
                        ranges_updated = True
                else:
                    updated_validated_answers[char_range] = count
            if ranges_updated:
                updated_validated_answers = json.dumps(
                    updated_validated_answers, ensure_ascii=False, separators=(",", ":")
                )
                dataset.at[row.Index, "validated_answers"] = updated_validated_answers

    # Process Splits
    data = []
    cache = dict()

    train_story_ids = set(
        pd.read_csv(os.path.join(task_data_path, "train_story_ids.csv"))["story_id"].values
    )
    dev_story_ids = set(
        pd.read_csv(os.path.join(task_data_path, "dev_story_ids.csv"))["story_id"].values
    )
    test_story_ids = set(
        pd.read_csv(os.path.join(task_data_path, "test_story_ids.csv"))["story_id"].values
    )

    def _get_data_type(story_id_):
        if story_id_ in train_story_ids:
            return "train"
        elif story_id_ in dev_story_ids:
            return "dev"
        elif story_id_ in test_story_ids:
            return "test"
        else:
            return ValueError("{} not found in any story ID set.".format(story_id))

    for row in display.tqdm(dataset.itertuples(), total=len(dataset), desc="Building json"):
        questions = cache.get(row.story_id)
        if questions is None:
            questions = []
            datum = dict(
                storyId=row.story_id,
                type=_get_data_type(row.story_id),
                text=row.story_text,
                questions=questions,
            )
            cache[row.story_id] = questions
            data.append(datum)
        q = dict(
            q=row.question,
            answers=_map_answers(row.answer_char_ranges),
            isAnswerAbsent=row.is_answer_absent,
        )
        if row.is_question_bad != "?":
            q["isQuestionBad"] = float(row.is_question_bad)
        if row.validated_answers and not pd.isnull(row.validated_answers):
            validated_answers = json.loads(row.validated_answers)
            q["validatedAnswers"] = []
            for answer, count in validated_answers.items():
                answer_item = dict(count=count)
                if answer == "none":
                    answer_item["noAnswer"] = True
                elif answer == "bad_question":
                    answer_item["badQuestion"] = True
                else:
                    s, e = map(int, answer.split(":"))
                    answer_item["s"] = s
                    answer_item["e"] = e
                q["validatedAnswers"].append(answer_item)
        consensus_start, consensus_end = get_consensus_answer(row)
        if consensus_start is None and consensus_end is None:
            if q.get("isQuestionBad", 0) >= 0.5:
                q["consensus"] = dict(badQuestion=True)
            else:
                q["consensus"] = dict(noAnswer=True)
        else:
            q["consensus"] = dict(s=consensus_start, e=consensus_end)
        questions.append(q)

    phase_dict = {
        "train": [],
        "val": [],
        "test": [],
    }
    phase_map = {"train": "train", "dev": "val", "test": "test"}
    for entry in data:
        phase = phase_map[entry["type"]]
        output_entry = {"text": entry["text"], "storyId": entry["storyId"], "qas": []}
        for qn in entry["questions"]:
            if "badQuestion" in qn["consensus"] or "noAnswer" in qn["consensus"]:
                continue
            output_entry["qas"].append({"question": qn["q"], "answer": qn["consensus"]})
        phase_dict[phase].append(output_entry)
    for phase, phase_data in phase_dict.items():
        py_io.write_jsonl(phase_data, os.path.join(task_data_path, f"{phase}.jsonl"))
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.jsonl"),
                "val": os.path.join(task_data_path, "val.jsonl"),
                "test": os.path.join(task_data_path, "val.jsonl"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )
    for file_name in file_name_list:
        os.remove(os.path.join(task_data_path, file_name))


def download_mrqa_natural_questions_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_file(
        "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz",
        os.path.join(task_data_path, "train.jsonl.gz"),
    )
    download_utils.download_file(
        "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz",
        os.path.join(task_data_path, "val.jsonl.gz"),
    )
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.jsonl.gz"),
                "val": os.path.join(task_data_path, "val.jsonl.gz"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_piqa_data_and_write_config(task_name: str, task_data_path: str, task_config_path: str):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_file(
        "https://yonatanbisk.com/piqa/data/train.jsonl",
        os.path.join(task_data_path, "train.jsonl"),
    )
    download_utils.download_file(
        "https://yonatanbisk.com/piqa/data/train-labels.lst",
        os.path.join(task_data_path, "train-labels.lst"),
    )
    download_utils.download_file(
        "https://yonatanbisk.com/piqa/data/valid.jsonl",
        os.path.join(task_data_path, "valid.jsonl"),
    )
    download_utils.download_file(
        "https://yonatanbisk.com/piqa/data/valid-labels.lst",
        os.path.join(task_data_path, "valid-labels.lst"),
    )
    download_utils.download_file(
        "https://yonatanbisk.com/piqa/data/tests.jsonl",
        os.path.join(task_data_path, "tests.jsonl"),
    )

    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.jsonl"),
                "train_labels": os.path.join(task_data_path, "train-labels.lst"),
                "val": os.path.join(task_data_path, "valid.jsonl"),
                "val_labels": os.path.join(task_data_path, "valid-labels.lst"),
                "test": os.path.join(task_data_path, "tests.jsonl"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_winogrande_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_unzip(
        "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip",
        task_data_path,
    )

    task_data_path = os.path.join(task_data_path, "winogrande_1.1")

    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train_xl.jsonl"),
                "train_labels": os.path.join(task_data_path, "train_xl-labels.lst"),
                "train_xs": os.path.join(task_data_path, "train_xs.jsonl"),
                "train_xs_labels": os.path.join(task_data_path, "train_xs-labels.lst"),
                "train_s": os.path.join(task_data_path, "train_s.jsonl"),
                "train_s_labels": os.path.join(task_data_path, "train_s-labels.lst"),
                "train_m": os.path.join(task_data_path, "train_m.jsonl"),
                "train_m_labels": os.path.join(task_data_path, "train_m-labels.lst"),
                "train_l": os.path.join(task_data_path, "train_l.jsonl"),
                "train_l_labels": os.path.join(task_data_path, "train_l-labels.lst"),
                "train_xl": os.path.join(task_data_path, "train_xl.jsonl"),
                "train_xl_labels": os.path.join(task_data_path, "train_xl-labels.lst"),
                "val": os.path.join(task_data_path, "dev.jsonl"),
                "val_labels": os.path.join(task_data_path, "dev-labels.lst"),
                "test": os.path.join(task_data_path, "test.jsonl"),
            },
            "name": task_name,
        },
        path=task_config_path,
    )


def download_ropes_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    os.makedirs(task_data_path, exist_ok=True)
    download_utils.download_and_untar(
        "https://ropes-dataset.s3-us-west-2.amazonaws.com/train_and_dev/"
        "ropes-train-dev-v1.0.tar.gz",
        task_data_path,
    )
    data_phase_list = ["train", "dev"]
    jiant_phase_list = ["train", "val"]
    for data_phase, jiant_phase in zip(data_phase_list, jiant_phase_list):
        os.rename(
            os.path.join(task_data_path, "ropes-train-dev-v1.0", f"{data_phase}-v1.0.json"),
            os.path.join(task_data_path, f"{jiant_phase}.json"),
        )
    shutil.rmtree(os.path.join(task_data_path, "ropes-train-dev-v1.0"))
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {
                "train": os.path.join(task_data_path, "train.json"),
                "val": os.path.join(task_data_path, "val.json"),
            },
            "name": task_name,
            "kwargs": {"include_background": True},
        },
        path=task_config_path,
    )


def download_acceptability_judgments_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    dataset_name = {
        "acceptability_definiteness": "definiteness",
        "acceptability_coord": "coordinating-conjunctions",
        "acceptability_whwords": "whwords",
        "acceptability_eos": "eos",
    }[task_name]
    os.makedirs(task_data_path, exist_ok=True)
    # data contains all train/val/test examples
    # metadata contains the split indicators
    # (there are 10 CV folds, we use fold1 by default, see below)
    data_path = os.path.join(task_data_path, "data.json")
    metadata_path = os.path.join(task_data_path, "metadata.json")
    download_utils.download_file(
        url="https://raw.githubusercontent.com/decompositional-semantics-initiative/DNC/master/"
        f"function_words/ACCEPTABILITY/acceptability-{dataset_name}_data.json",
        file_path=data_path,
    )
    download_utils.download_file(
        url="https://raw.githubusercontent.com/decompositional-semantics-initiative/DNC/master/"
        f"function_words/ACCEPTABILITY/acceptability-{dataset_name}_metadata.json",
        file_path=metadata_path,
    )
    py_io.write_json(
        data={
            "task": task_name,
            "paths": {"data": data_path, "metadata": metadata_path},
            "name": task_name,
            "kwargs": {"fold": "fold1"},  # use fold1 (out of 10) by default
        },
        path=task_config_path,
    )


def download_senteval_data_and_write_config(
    task_name: str, task_data_path: str, task_config_path: str
):
    name_map = {
        "senteval_bigram_shift": "bigram_shift",
        "senteval_coordination_inversion": "coordination_inversion",
        "senteval_obj_number": "obj_number",
        "senteval_odd_man_out": "odd_man_out",
        "senteval_past_present": "past_present",
        "senteval_sentence_length": "sentence_length",
        "senteval_subj_number": "subj_number",
        "senteval_top_constituents": "top_constituents",
        "senteval_tree_depth": "tree_depth",
        "senteval_word_content": "word_content",
    }
    dataset_name = name_map[task_name]
    os.makedirs(task_data_path, exist_ok=True)
    # data contains all train/val/test examples, first column indicates the split
    data_path = os.path.join(task_data_path, "data.tsv")
    download_utils.download_file(
        url="https://raw.githubusercontent.com/facebookresearch/SentEval/master/data/probing/"
        f"{dataset_name}.txt",
        file_path=data_path,
    )
    py_io.write_json(
        data={"task": task_name, "paths": {"data": data_path}, "name": task_name},
        path=task_config_path,
    )
