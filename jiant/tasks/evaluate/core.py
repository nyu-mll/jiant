import json
import os
import torch

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from typing import Dict

import jiant.tasks as tasks
from jiant.utils.python.datastructures import ExtendedDataClassMixin


@dataclass
class Metrics(ExtendedDataClassMixin):
    major: float
    minor: Dict


class BaseEvaluation:
    pass


class BaseAccumulator:
    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        raise NotImplementedError()

    def get_accumulated(self):
        raise NotImplementedError()


class BaseEvaluationScheme:
    def get_accumulator(self) -> BaseAccumulator:
        raise NotImplementedError()

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        # Depending on the task, labels may be more easily extracted from
        #   a cache or raw examples.
        # Provide the EvaluationScheme with either, but delegate to another function
        #   using only one.
        raise NotImplementedError()

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError()

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:
        raise NotImplementedError()


class ConcatenateLogitsAccumulator(BaseAccumulator):
    def __init__(self):
        self.logits_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        self.logits_list.append(batch_logits)

    def get_accumulated(self):
        all_logits = np.concatenate(self.logits_list)
        return all_logits


class ConcatenateLossAccumulator(BaseAccumulator):
    def __init__(self):
        self.loss_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        self.loss_list.append(batch_loss)

    def get_accumulated(self):
        all_loss = np.array(self.loss_list)
        return all_loss


class BaseLogitsEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return get_label_ids_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError()

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels,)

    def compute_metrics_from_preds_and_labels(self, preds, labels):
        raise NotImplementedError()


class SimpleAccuracyEvaluationScheme(BaseLogitsEvaluationScheme):
    @classmethod
    def get_preds_from_accumulator(cls, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        # noinspection PyUnresolvedReferences
        acc = float((preds == labels).mean())
        return Metrics(major=acc, minor={"acc": acc},)


class AccAndF1EvaluationScheme(BaseLogitsEvaluationScheme):
    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        # noinspection PyUnresolvedReferences
        acc = float((preds == labels).mean())
        labels = np.array(labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        minor = {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }
        return Metrics(major=minor["acc_and_f1"], minor=minor,)


class MCCEvaluationScheme(BaseLogitsEvaluationScheme):
    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        mcc = matthews_corrcoef(labels, preds)
        return Metrics(major=mcc, minor={"mcc": mcc},)


class PearsonAndSpearmanEvaluationScheme(BaseLogitsEvaluationScheme):
    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return get_label_vals_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.squeeze(logits, axis=-1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        pearson_corr = float(pearsonr(preds, labels)[0])
        spearman_corr = float(spearmanr(preds, labels)[0])
        minor = {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
        return Metrics(major=minor["corr"], minor=minor,)


def get_evaluation_scheme_for_task(task) -> BaseEvaluationScheme:
    # Todo: move logic to task?
    if isinstance(
        task,
        (
            tasks.MnliTask,
            tasks.QnliTask,
            tasks.RteTask,
            tasks.SnliTask,
            tasks.SstTask,
            tasks.WnliTask,
        ),
    ):
        return SimpleAccuracyEvaluationScheme()
    elif isinstance(task, tasks.ColaTask):
        return MCCEvaluationScheme()
    elif isinstance(task, (tasks.MrpcTask, tasks.QqpTask,)):
        return AccAndF1EvaluationScheme()
    elif isinstance(task, tasks.StsbTask):
        return PearsonAndSpearmanEvaluationScheme()
    else:
        raise KeyError(task)


def get_label_ids(task, examples):
    return np.array([task.LABEL_BIMAP.a[example.label] for example in examples])


def get_label_id_from_data_row(data_row):
    return data_row.label_id


def get_label_ids_from_cache(cache):
    return np.array(
        [get_label_id_from_data_row(data_row=datum["data_row"]) for datum in cache.iter_all()]
    )


def get_label_vals_from_cache(cache):
    return np.array(
        [get_label_val_from_data_row(data_row=datum["data_row"]) for datum in cache.iter_all()]
    )


def get_label_val_from_data_row(data_row):
    return data_row.label


def mean(*args) -> float:
    return float(np.mean(args))


def write_metrics(results, output_path, verbose=True):
    results_to_write = {}
    if "loss" in results:
        results_to_write["loss"] = results["loss"]
    if "metrics" in results:
        results_to_write["metrics"] = results["metrics"].asdict()
    assert results_to_write
    metrics_str = json.dumps(results_to_write, indent=2)
    if verbose:
        print(metrics_str)
    with open(output_path, "w") as f:
        f.write(metrics_str)


def write_preds(logits, output_path):
    df = pd.DataFrame(logits)
    df.to_csv(output_path, header=False, index=False)


def write_val_results(results, output_dir, verbose=True, do_write_preds=True):
    os.makedirs(output_dir, exist_ok=True)
    if do_write_preds:
        if len(results["logits"].shape) == 2:
            write_preds(
                logits=results["logits"], output_path=os.path.join(output_dir, "val_preds.csv"),
            )
        else:
            torch.save(results["logits"], os.path.join(output_dir, "val_preds.p"))
    write_metrics(
        results=results, output_path=os.path.join(output_dir, "val_metrics.json"), verbose=verbose,
    )
