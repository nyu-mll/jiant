import os
import csv
import torch

import jiant.utils.python.io as py_io
from jiant.tasks import retrieval
from jiant.tasks.constants import GLUE_TASKS, SUPERGLUE_TASKS


class Benchmark:
    TASKS = NotImplemented
    BENCHMARK_SUBMISSION_FILENAMES = NotImplemented

    @classmethod
    def write_predictions(cls, task_name: str, input_filepath: str, output_filepath: str):
        raise NotImplementedError


# https://gluebenchmark.com/
class GlueBenchmark(Benchmark):
    TASKS = GLUE_TASKS
    BENCHMARK_SUBMISSION_FILENAMES = {
        "cola": "CoLA.tsv",
        "sst": "SST-2.tsv",
        "mrpc": "MRPC.tsv",
        "stsb": "STS-B.tsv",
        "mnli": "MNLI-m.tsv",
        "mnli_mismatched": "MNLI-mm.tsv",
        "qnli": "QNLI.tsv",
        "qqp": "QQP.tsv",
        "rte": "RTE.tsv",
        "wnli": "WNLI.tsv",
        "glue_diagnostics": "AX.tsv",
    }

    @classmethod
    def write_predictions(cls, task_name: str, input_filepath: str, output_filepath: str):
        task = retrieval.get_task_class(task_name)
        task_preds = torch.load(input_filepath)[task_name]
        indexes, predictions = task.get_glue_preds(task_preds)
        with open(output_filepath, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(("index", "prediction"))
            writer.writerows(zip(indexes, predictions))


# https://super.gluebenchmark.com/
class SuperglueBenchmark(Benchmark):
    TASKS = SUPERGLUE_TASKS
    BENCHMARK_SUBMISSION_FILENAMES = {
        "boolq": "BoolQ.jsonl",
        "cb": "CB.jsonl",
        "copa": "COPA.jsonl",
        "multirc": "MultiRC.jsonl",
        "record": "ReCoRD.jsonl",
        "rte": "RTE.jsonl",
        "wic": "WiC.jsonl",
        "wsc": "WSC.jsonl",
        "superglue_axb": "AX-b.jsonl",
        "superglue_axg": "AX-g.jsonl",
    }

    @classmethod
    def write_predictions(cls, task_name: str, input_filepath: str, output_filepath: str):
        task = retrieval.get_task_class(task_name)
        task_preds = torch.load(input_filepath)[task_name]
        formatted_preds = task.super_glue_format_preds(task_preds)
        py_io.write_jsonl(
            data=formatted_preds,
            path=os.path.join(
                SuperglueBenchmark.BENCHMARK_SUBMISSION_FILENAMES[task_name], output_filepath
            ),
        )
