import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seqeval.metrics as seqeval_metrics
import torch
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List

import jiant.shared.model_resolution as model_resolution
import jiant.tasks as tasks
import jiant.tasks.lib.templates.squad_style.core as squad_style
import jiant.tasks.lib.templates.squad_style.utils as squad_style_utils
import jiant.tasks.lib.mlqa as mlqa_lib
import jiant.tasks.lib.bucc2018 as bucc2018_lib
import jiant.tasks.lib.tatoeba as tatoeba_lib
from jiant.tasks.lib.templates import mlm as mlm_template
from jiant.utils.python.datastructures import ExtendedDataClassMixin
from jiant.utils.python.io import read_json
from jiant.utils.string_comparing import string_f1_score, exact_match_score


@dataclass
class Metrics(ExtendedDataClassMixin):
    major: float
    minor: Dict


class BaseEvaluation:
    pass


class BaseAccumulator:
    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        raise NotImplementedError()

    def get_guids(self):
        return None

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
        self.guid_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        self.logits_list.append(batch_logits)
        batch_guid = batch_metadata.get("guid")
        if batch_guid is not None:
            self.guid_list.append(batch_guid)

    def get_guids(self):
        if self.guid_list:
            return np.concatenate(self.guid_list)
        else:
            return None

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


class ConcatenateStringListAccumulator(BaseAccumulator):
    def __init__(self):
        self.str_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        bs = len(batch_logits)
        span_pred = batch_logits.argmax(axis=1)
        pred_token_start, pred_token_end = span_pred[:, 0], span_pred[:, 1]
        pred_char_start = batch.token_idx_to_char_idx_start.cpu().numpy()[
            range(bs), pred_token_start
        ]
        pred_char_end = batch.token_idx_to_char_idx_end.cpu().numpy()[range(bs), pred_token_end]
        self.str_list.extend(
            [
                s[i1 : i2 + 1]
                for i1, i2, s in zip(pred_char_start, pred_char_end, batch.selection_str)
            ]
        )

    def get_accumulated(self):
        return self.str_list


class SpanPredictionF1andEMScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateStringListAccumulator()

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return [datum["data_row"].gt_span_str for datum in cache.iter_all()]

    def get_preds_from_accumulator(self, task, accumulator):
        return accumulator.get_accumulated()

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        em = sum([exact_match_score(s1, s2) for s1, s2 in zip(preds, labels)]) / len(labels)
        f1 = sum([string_f1_score(s1, s2) for s1, s2 in zip(preds, labels)]) / len(labels)
        scores = {"f1": f1, "em": em, "avg": (f1 + em) / 2}
        return Metrics(major=scores["avg"], minor=scores)

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateStringListAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels)


class RecordAccumulator(ConcatenateLogitsAccumulator):
    def __init__(self):
        super().__init__()
        self.entity_strs = []
        self.gold_label_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        super().update(batch_logits, batch_loss, batch, batch_metadata)
        self.entity_strs.extend(batch.entity_str)
        self.gold_label_list.extend(batch.label_set)

    def get_accumulated(self):
        return super().get_accumulated(), self.entity_strs

    def get_gold_label_list(self):
        return self.gold_label_list


class MLMPremaskedAccumulator(BaseAccumulator):
    def __init__(self):
        self.loss_list = []
        self.logits_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        batch_size = len(batch)
        # Select the tokens that we do MLM prediction on
        masked_tokens_selector = (
            batch.masked_lm_labels.cpu().numpy() != mlm_template.NON_MASKED_TOKEN_LABEL_ID
        )
        for i in range(batch_size):
            # noinspection PyUnresolvedReferences
            self.logits_list.append(batch_logits[i][masked_tokens_selector[i]])
        self.loss_list.append(batch_loss)

    def get_accumulated(self):
        return self.loss_list, self.logits_list


class TatoebaAccumulator(BaseAccumulator):
    def __init__(self):
        self.embeddings_list = []
        self.is_english_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        self.embeddings_list.append(batch_logits)
        self.is_english_list.append(batch.is_english.cpu().numpy())

    @classmethod
    def get_guids(cls):
        return None

    def get_accumulated(self):
        all_embeddings = np.concatenate(self.embeddings_list)
        is_english_arr = np.concatenate(self.is_english_list).astype(bool)
        return all_embeddings, is_english_arr


class Bucc2018Accumulator(BaseAccumulator):
    def __init__(self):
        self.embeddings_list = []
        self.is_english_list = []
        self.text_hash_list = []
        self.guid_list = []

    def update(self, batch_logits, batch_loss, batch, batch_metadata):
        self.embeddings_list.append(batch_logits)
        self.is_english_list.append(batch.is_english.cpu().numpy())
        self.text_hash_list += batch.text_hash
        self.guid_list += batch.guid

    @classmethod
    def get_guids(cls):
        return None

    def get_accumulated(self):
        return {
            "all_embeddings": np.concatenate(self.embeddings_list),
            "is_english_arr": np.concatenate(self.is_english_list).astype(bool),
            "text_hash_list": self.text_hash_list,
            "guid_list": self.guid_list,
        }


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
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels)

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
        return Metrics(major=acc, minor={"acc": acc})


class MCTACOEvaluationScheme(BaseLogitsEvaluationScheme):
    @classmethod
    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        pred = np.argmax(logits, axis=1)
        guid = accumulator.get_guids()
        return guid, pred

    @classmethod
    def compute_metrics_from_accumulator(self, task, accumulator, tokenizer, labels) -> Metrics:
        guid, pred = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        em_ls = []
        f1_ls = []
        label_pred_by_question = {}

        for one_guid, one_pred, one_label in zip(guid, pred, labels):
            split, question_id, example_id = one_guid.split("-")
            if question_id not in label_pred_by_question:
                label_pred_by_question[question_id] = [], []
            label_pred_by_question[question_id][0].append(one_label)
            label_pred_by_question[question_id][1].append(one_pred)

        em_ls = [
            float(group_label == group_pred)
            for group_label, group_pred in label_pred_by_question.values()
        ]
        f1_ls = [
            f1_score(y_true=group_label, y_pred=group_pred)
            for group_label, group_pred in label_pred_by_question.values()
        ]

        em = sum(em_ls) / len(em_ls)
        f1 = sum(f1_ls) / len(f1_ls)
        minor = {
            "em": em,
            "f1": f1,
            "f1_em": (f1 + em) / 2,
        }
        metrics = Metrics(major=minor["f1_em"], minor=minor,)
        return metrics


class MultiLabelAccAndF1EvaluationScheme(BaseLogitsEvaluationScheme):
    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return get_multi_label_ids_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return (logits > 0.5).astype(int)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        # noinspection PyUnresolvedReferences
        acc = float((preds == labels).mean())
        labels = np.array(labels)
        minor = {
            "acc": acc,
            "f1_micro": f1_score(y_true=labels, y_pred=preds, average="micro"),
            "acc_and_f1_micro": (acc + f1_score(y_true=labels, y_pred=preds, average="micro")) / 2,
        }
        return Metrics(major=minor["acc_and_f1_micro"], minor=minor)


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
        return Metrics(major=minor["acc_and_f1"], minor=minor)


class MCCEvaluationScheme(BaseLogitsEvaluationScheme):
    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        mcc = matthews_corrcoef(labels, preds)
        return Metrics(major=mcc, minor={"mcc": mcc})


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
        return Metrics(major=minor["corr"], minor=minor)


class MultipleChoiceAccuracyEvaluationScheme(BaseLogitsEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_examples(cls, task, examples):
        return get_multiple_choice_label_ids_from_examples(task=task, examples=examples)

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return get_multiple_choice_labels_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        return SimpleAccuracyEvaluationScheme.get_preds_from_accumulator(
            task=task, accumulator=accumulator,
        )

    def compute_metrics_from_preds_and_labels(self, preds, labels):
        return SimpleAccuracyEvaluationScheme.compute_metrics_from_preds_and_labels(
            preds=preds, labels=labels
        )


class CommitmentBankEvaluationScheme(BaseLogitsEvaluationScheme):
    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=1)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        # noinspection PyUnresolvedReferences
        acc = float((preds == labels).mean())
        labels = np.array(labels)
        f11 = f1_score(y_true=labels == 0, y_pred=preds == 0)
        f12 = f1_score(y_true=labels == 1, y_pred=preds == 1)
        f13 = f1_score(y_true=labels == 2, y_pred=preds == 2)
        avg_f1 = mean(f11, f12, f13)
        return Metrics(
            major=mean(acc, avg_f1),
            minor={"acc": acc, "avg_f1": avg_f1, "f11": f11, "f12": f12, "f13": f13},
        )


class MultiRCEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_examples(cls, task, examples):
        label_values = get_label_ids(examples=examples, task=task)
        question_ids = np.array([example.question_id for example in examples])
        assert len(label_values) == len(question_ids)
        return [
            {"label_values": lab, "question_ids": qid}
            for lab, qid in zip(label_values, question_ids)
        ]

    @classmethod
    def get_labels_from_cache(cls, cache):
        label_values = []
        question_ids = []
        for datum in cache.iter_all():
            label_values.append(datum["data_row"].label_id)
            question_ids.append(datum["data_row"].question_id)
        label_values = np.array(label_values)
        question_ids = np.array(question_ids)
        assert len(label_values) == len(question_ids)
        return [
            {"label_values": lab, "question_ids": qid}
            for lab, qid in zip(label_values, question_ids)
        ]

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return self.get_labels_from_examples(task=task, examples=examples)

    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=-1)

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels,)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        df = pd.DataFrame(labels)
        assert "label_values" in df.columns
        assert "question_ids" in df.columns
        df["preds"] = preds
        # noinspection PyUnresolvedReferences
        exact_match = (
            df.groupby("question_ids")
            .apply(lambda _: (_["preds"] == _["label_values"]).all())
            .mean()
        )
        exact_match = float(exact_match)
        f1 = f1_score(y_true=df["label_values"], y_pred=df["preds"])
        return Metrics(major=mean(exact_match, f1), minor={"em": exact_match, "f1": f1},)


@dataclass
class RecordLabelData:
    passage_idx: int
    question_idx: int
    entity_str: str
    answers_dict: Dict[str, str]


class ReCordEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return RecordAccumulator()

    @classmethod
    def get_labels_from_examples(cls, examples):
        return [
            RecordLabelData(
                passage_idx=example.passage_idx,
                question_idx=example.question_idx,
                entity_str=example.entity_str,
                answers_dict=example.answers_dict,
            )
            for example in examples
        ]

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        return cls.get_labels_from_examples(examples=examples)

    @classmethod
    def get_preds_from_accumulator(cls, task, accumulator):
        logits, entity_strs = accumulator.get_accumulated()
        guid_list = accumulator.get_guids()

        question_ids = []
        for guid in guid_list:
            question_ids.append(guid.split("-")[2])

        # group logits by question id then reorder for submission
        # need question id, logit, and entity_str
        # for example, dict of question id to logit and entity_str
        max_logits = {}
        for logit, entity_str, question_id in zip(logits, entity_strs, question_ids):
            if (question_id not in max_logits) or (max_logits[question_id]["logit"][1] < logit[1]):
                max_logits[question_id] = {"logit": logit, "entity_str": entity_str}

        # Convert labels of max_logits to prediction format
        preds = []
        for question_idx, logit_entity in max_logits.items():
            preds.append({"idx": question_idx, "label": logit_entity["entity_str"]})

        return preds

    def compute_metrics_from_accumulator(
        self, task, accumulator: RecordAccumulator, tokenizer, labels: List
    ) -> Metrics:
        predictions_dict, metrics = self.compute_preds_and_metrics(task, accumulator)
        return metrics

    @classmethod
    def compute_preds_and_metrics(cls, task, accumulator):
        f1_ls = []
        em_ls = []
        predictions_dict = {}

        preds = cls.get_preds_from_accumulator(task, accumulator)
        guid_list = accumulator.get_guids()
        gold_label_list_of_sets = accumulator.get_gold_label_list()

        question_ids = []
        for guid in guid_list:
            question_ids.append(guid.split("-")[2])

        # Reduce list of gold label sets to a gold label set per question_id
        gold_labels = {}
        for question_id, gold_label_set in zip(question_ids, gold_label_list_of_sets):
            if question_id in gold_labels:
                assert gold_label_set == gold_labels[question_id]
            else:
                gold_labels[question_id] = gold_label_set

        for pred, gold_label_set in zip(preds, gold_labels.values()):
            pred_ans = pred["label"]

            # F1
            f1 = cls.metric_max_over_ground_truths(string_f1_score, pred_ans, gold_label_set)
            f1_ls.append(f1)

            # EM
            em = cls.metric_max_over_ground_truths(exact_match_score, pred_ans, gold_label_set)
            em_ls.append(em)

        em = sum(em_ls) / len(em_ls)
        f1 = sum(f1_ls) / len(f1_ls)
        minor = {
            "em": em,
            "f1": f1,
            "f1_em": (f1 + em) / 2,
        }
        metrics = Metrics(major=minor["f1_em"], minor=minor,)
        return predictions_dict, metrics

    @classmethod
    def metric_max_over_ground_truths(cls, metric_fn, prediction, ground_truths):
        """Compute max metric between prediction and each ground truth.
        From official ReCoRD eval script
        """
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)


class CCGEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_label_ids_from_cache(cls, cache):
        return [
            {"label_ids": datum["data_row"].label_ids, "label_mask": datum["data_row"].label_mask}
            for datum in cache.iter_all()
        ]

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        return cls.get_label_ids_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=-1)

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels,)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        label_ids = np.stack([row["label_ids"] for row in labels])
        label_mask = np.stack([row["label_mask"] for row in labels])

        # Account for smart-truncate
        assert (label_mask[:, preds.shape[-1] :] == 0).all()
        label_ids = label_ids[:, : preds.shape[-1]]
        label_mask = label_mask[:, : preds.shape[-1]]

        bool_mask = label_mask.reshape(-1).astype(bool)
        flat_preds = preds.reshape(-1)[bool_mask]
        flat_labels = label_ids.reshape(-1)[bool_mask]
        return cls.compute_metrics_from_flat_preds_and_labels(
            flat_preds=flat_preds, flat_labels=flat_labels,
        )

    @classmethod
    def compute_metrics_from_flat_preds_and_labels(cls, flat_preds, flat_labels):
        return SimpleAccuracyEvaluationScheme.compute_metrics_from_preds_and_labels(
            preds=flat_preds, labels=flat_labels,
        )


class F1TaggingEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        labels = []
        for datum in cache.iter_all():
            label_mask = datum["data_row"].label_mask.astype(bool)
            pos_list = [
                task.ID_TO_LABEL[pos_id] for pos_id in datum["data_row"].label_ids[label_mask]
            ]
            label = {
                "pos_list": pos_list,
                "label_mask": label_mask,
            }
            labels.append(label)
            assert len(pos_list) == label_mask.sum()
        return labels

    def get_preds_from_accumulator(self, task, accumulator):
        logits = accumulator.get_accumulated()
        return np.argmax(logits, axis=-1)

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(task=task, preds=preds, labels=labels,)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, task, preds, labels):
        label_mask = np.stack([row["label_mask"] for row in labels])

        # Account for smart-truncate
        assert (label_mask[:, preds.shape[-1] :] == 0).all()
        label_mask = label_mask[:, : preds.shape[-1]]

        labels_for_eval = [label["pos_list"] for label in labels]
        preds_for_eval = []
        assert len(labels) == preds.shape[0]
        for i in range(len(labels)):
            relevant_preds = preds[i][label_mask[i]]
            relevant_preds_pos = [task.ID_TO_LABEL[pos_id] for pos_id in relevant_preds]
            preds_for_eval.append(relevant_preds_pos)

        minor = {
            "precision": seqeval_metrics.precision_score(labels_for_eval, preds_for_eval),
            "recall": seqeval_metrics.recall_score(labels_for_eval, preds_for_eval),
            "f1": seqeval_metrics.f1_score(labels_for_eval, preds_for_eval),
        }
        return Metrics(major=minor["f1"], minor=minor,)


class SQuADEvaluationScheme(BaseEvaluationScheme):
    @classmethod
    def get_accumulator(cls) -> BaseAccumulator:
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_cache(cls, cache):
        return [cls.get_label_from_data_row(datum["data_row"]) for datum in cache.iter_all()]

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        return cls.get_labels_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError("Currently can't be done without access to dataset")

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:
        logits = accumulator.get_accumulated()
        results, predictions = squad_style.compute_predictions_logits_v3(
            data_rows=labels,
            logits=logits,
            n_best_size=task.n_best_size,
            max_answer_length=task.max_answer_length,
            do_lower_case=model_resolution.resolve_is_lower_case(tokenizer),
            version_2_with_negative=task.version_2_with_negative,
            null_score_diff_threshold=task.null_score_diff_threshold,
            tokenizer=tokenizer,
        )
        if task.version_2_with_negative:
            # Return the score after the best thresholds for answering has been selected
            return Metrics(major=(results["best_f1"] + results["best_exact"]) / 2, minor=results)
        else:
            return Metrics(major=(results["f1"] + results["exact"]) / 2, minor=results)

    @classmethod
    def get_label_from_data_row(cls, data_row):
        return squad_style.PartialDataRow.from_data_row(data_row)


class XlingQAEvaluationScheme(BaseEvaluationScheme):
    @classmethod
    def get_accumulator(cls) -> BaseAccumulator:
        return ConcatenateLogitsAccumulator()

    @classmethod
    def get_labels_from_cache(cls, cache):
        return [cls.get_label_from_data_row(datum["data_row"]) for datum in cache.iter_all()]

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        return cls.get_labels_from_cache(cache=cache)

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError("Currently can't be done without access to dataset")

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:
        logits = accumulator.get_accumulated()
        assert isinstance(task, (tasks.TyDiQATask, tasks.XquadTask))
        lang = task.language
        results, predictions = squad_style.compute_predictions_logits_v3(
            data_rows=labels,
            logits=logits,
            n_best_size=task.n_best_size,
            max_answer_length=task.max_answer_length,
            do_lower_case=model_resolution.resolve_is_lower_case(tokenizer),
            version_2_with_negative=task.version_2_with_negative,
            null_score_diff_threshold=task.null_score_diff_threshold,
            skip_get_final_text=(lang == "zh"),
            tokenizer=tokenizer,
        )
        return Metrics(major=(results["f1"] + results["exact"]) / 2, minor=results,)

    @classmethod
    def get_label_from_data_row(cls, data_row):
        return squad_style.PartialDataRow.from_data_row(data_row)


class MLQAEvaluationScheme(SQuADEvaluationScheme):
    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError("Too hard for now, too much handled in one giant lib")

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:

        # Todo: Fix val labels cache
        # This is a quick hack
        logits = accumulator.get_accumulated()
        partial_examples = squad_style.data_rows_to_partial_examples(data_rows=labels)
        all_pred_results = squad_style.logits_to_pred_results_list(logits)
        assert task.context_language == task.question_language
        lang = task.context_language
        predictions = squad_style_utils.compute_predictions_logits_v2(
            partial_examples=partial_examples,
            all_results=all_pred_results,
            n_best_size=task.n_best_size,
            max_answer_length=task.max_answer_length,
            do_lower_case=model_resolution.resolve_is_lower_case(tokenizer),
            version_2_with_negative=task.version_2_with_negative,
            null_score_diff_threshold=task.null_score_diff_threshold,
            tokenizer=tokenizer,
            skip_get_final_text=(lang == "zh"),
            verbose=True,
        )
        dataset = read_json(task.val_path)["data"]
        results = mlqa_lib.evaluate(dataset=dataset, predictions=predictions, lang=lang,)
        return Metrics(major=(results["f1"] + results["exact_match"]) / 2, minor=results,)


class MLMEvaluationScheme(BaseEvaluationScheme):
    @classmethod
    def get_accumulator(cls) -> BaseAccumulator:
        return ConcatenateLossAccumulator()

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        # This is a dummy function. There are no external labels.
        return [None]

    def get_preds_from_accumulator(self, task, accumulator):
        raise NotImplementedError("Not possible")

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:
        loss_list = accumulator.get_accumulated()
        average_loss = mean(loss_list)
        perplexity = np.exp(average_loss)
        return Metrics(
            # Major = negative perplexity
            major=-perplexity,
            minor={"perplexity": perplexity},
        )


class MLMPremaskedEvaluationScheme(MLMEvaluationScheme):
    @classmethod
    def get_accumulator(cls) -> BaseAccumulator:
        return MLMPremaskedAccumulator()

    @classmethod
    def get_labels_from_cache_and_examples(cls, task, cache, examples):
        labels = []
        for datum in cache.iter_all():
            masked_lm_labels = datum["data_row"].masked_lm_labels
            labels.append(
                masked_lm_labels[masked_lm_labels != mlm_template.NON_MASKED_TOKEN_LABEL_ID]
            )
        return labels

    def get_preds_from_accumulator(self, task, accumulator):
        _, preds = accumulator.get_accumulated()
        return preds

    def compute_metrics_from_accumulator(
        self, task, accumulator: BaseAccumulator, tokenizer, labels
    ) -> Metrics:
        loss_list, _ = accumulator.get_accumulated()
        average_loss = mean(loss_list)
        perplexity = np.exp(average_loss)
        return Metrics(
            # Major = negative perplexity
            major=-perplexity,
            minor={"perplexity": perplexity},
        )


class TatoebaEvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return TatoebaAccumulator()

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return task.get_val_labels()

    def get_preds_from_accumulator(self, task, accumulator):
        all_embeddings, is_english_arr = accumulator.get_accumulated()
        other_lang_embeddings = all_embeddings[~is_english_arr]
        eng_embeddings = all_embeddings[is_english_arr]
        predictions = tatoeba_lib.similarity_search(
            x=other_lang_embeddings,
            y=eng_embeddings,
            dim=other_lang_embeddings.shape[-1],
            normalize=True,
        ).flatten()
        return predictions

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels,)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        # noinspection PyUnresolvedReferences
        acc = (preds == labels).mean()
        return Metrics(major=acc, minor={"acc": acc})


class Bucc2018EvaluationScheme(BaseEvaluationScheme):
    def get_accumulator(self):
        return Bucc2018Accumulator()

    def get_labels_from_cache_and_examples(self, task, cache, examples):
        return task.get_val_labels()

    def get_preds_from_accumulator(self, task, accumulator, threshold=0):
        accumulated = accumulator.get_accumulated()
        is_english_arr = accumulated["is_english_arr"]
        all_embeddings = accumulated["all_embeddings"]
        guids = accumulated["guid_list"]
        text_hash_list = accumulated["text_hash_list"]
        other_lang_embeddings = all_embeddings[~is_english_arr]
        eng_embeddings = all_embeddings[is_english_arr]
        english_guids = [x.split("-", 1)[1] for x in np.array(guids)[is_english_arr]]
        other_guids = [x.split("-", 1)[1] for x in np.array(guids)[~is_english_arr]]

        n = len(is_english_arr)
        src_inds, _ = bucc2018_lib.get_unique_lines(
            [text_hash_list[i] for i in np.arange(n) if not is_english_arr[i]]
        )
        trg_inds, _ = bucc2018_lib.get_unique_lines(
            [text_hash_list[i] for i in np.arange(n) if is_english_arr[i]]
        )
        src_ids_map = bucc2018_lib.create_ids_map(src_inds, other_guids)
        trg_ids_map = bucc2018_lib.create_ids_map(trg_inds, english_guids)

        result = bucc2018_lib.mine_bitext(
            x=other_lang_embeddings,
            y=eng_embeddings,
            src_inds=src_inds,
            trg_inds=trg_inds,
            threshold=threshold,
            use_gpu=torch.cuda.is_available(),
        )
        # Note: Setting thresholds only available in test script
        candidates2score = {}
        for score, src_idx, trg_idx in result:
            for src_key, trg_key in itertools.product(src_ids_map[src_idx], trg_ids_map[trg_idx]):
                candidates2score[src_key, trg_key] = score
        return candidates2score

    def compute_metrics_from_accumulator(
        self, task, accumulator: ConcatenateLogitsAccumulator, tokenizer, labels: list
    ) -> Metrics:
        preds = self.get_preds_from_accumulator(task=task, accumulator=accumulator)
        return self.compute_metrics_from_preds_and_labels(preds=preds, labels=labels,)

    @classmethod
    def compute_metrics_from_preds_and_labels(cls, preds, labels):
        labels = [tuple(x.split("\t")) for x in labels]
        result = bucc2018_lib.bucc_eval(preds, gold=labels, threshold=None)
        return Metrics(major=result["F1"], minor=result,)


def get_evaluation_scheme_for_task(task) -> BaseEvaluationScheme:
    # TODO: move logic to task?  (issue #1182)
    if isinstance(
        task,
        (
            tasks.AdversarialNliTask,
            tasks.AbductiveNliTask,
            tasks.AcceptabilityDefinitenessTask,
            tasks.BoolQTask,
            tasks.CopaTask,
            tasks.FeverNliTask,
            tasks.MnliTask,
            tasks.PawsXTask,
            tasks.QnliTask,
            tasks.RteTask,
            tasks.SciTailTask,
            tasks.SentevalTenseTask,
            tasks.SnliTask,
            tasks.SstTask,
            tasks.WiCTask,
            tasks.WnliTask,
            tasks.WSCTask,
            tasks.XnliTask,
            tasks.MCScriptTask,
            tasks.ArctTask,
            tasks.PiqaTask,
        ),
    ):
        return SimpleAccuracyEvaluationScheme()
    elif isinstance(task, tasks.MCTACOTask):
        return MCTACOEvaluationScheme()
    elif isinstance(task, tasks.CCGTask):
        return CCGEvaluationScheme()
    elif isinstance(task, tasks.CommitmentBankTask):
        return CommitmentBankEvaluationScheme()
    elif isinstance(task, tasks.ColaTask):
        return MCCEvaluationScheme()
    elif isinstance(
        task,
        (
            tasks.ArcEasyTask,
            tasks.ArcChallengeTask,
            tasks.CommonsenseQATask,
            tasks.CosmosQATask,
            tasks.SWAGTask,
            tasks.HellaSwagTask,
            tasks.MutualTask,
            tasks.MutualPlusTask,
            tasks.QuailTask,
            tasks.SocialIQATask,
            tasks.WinograndeTask,
            tasks.MCTestTask,
        ),
    ):
        return MultipleChoiceAccuracyEvaluationScheme()
    elif isinstance(task, (tasks.MrpcTask, tasks.QqpTask)):
        return AccAndF1EvaluationScheme()
    elif isinstance(
        task,
        (
            tasks.Spr1Task,
            tasks.Spr2Task,
            tasks.SemevalTask,
            tasks.SrlTask,
            tasks.NerTask,
            tasks.CorefTask,
            tasks.DprTask,
            tasks.DepTask,
            tasks.PosTask,
            tasks.NonterminalTask,
        ),
    ):
        return MultiLabelAccAndF1EvaluationScheme()
    elif isinstance(task, tasks.ReCoRDTask):
        return ReCordEvaluationScheme()
    elif isinstance(
        task, (tasks.SquadTask, tasks.QuorefTask, tasks.NewsQATask, tasks.MrqaNaturalQuestionsTask,)
    ):
        return SQuADEvaluationScheme()
    elif isinstance(task, (tasks.TyDiQATask, tasks.XquadTask)):
        return XlingQAEvaluationScheme()
    elif isinstance(task, tasks.MlqaTask):
        return MLQAEvaluationScheme()
    elif isinstance(task, tasks.MultiRCTask):
        return MultiRCEvaluationScheme()
    elif isinstance(task, tasks.StsbTask):
        return PearsonAndSpearmanEvaluationScheme()
    elif isinstance(task, tasks.MLMSimpleTask):
        return MLMEvaluationScheme()
    elif isinstance(task, (tasks.MLMPremaskedTask, tasks.MLMPretokenizedTask)):
        return MLMPremaskedEvaluationScheme()
    elif isinstance(task, (tasks.QAMRTask, tasks.QASRLTask)):
        return SpanPredictionF1andEMScheme()
    elif isinstance(task, (tasks.UdposTask, tasks.PanxTask)):
        return F1TaggingEvaluationScheme()
    elif isinstance(task, tasks.Bucc2018Task):
        return Bucc2018EvaluationScheme()
    elif isinstance(task, tasks.TatoebaTask):
        return TatoebaEvaluationScheme()
    else:
        raise KeyError(task)


def get_label_ids(task, examples):
    return np.array([task.LABEL_TO_ID[example.label] for example in examples])


def get_label_ids_from_data_row(data_row):
    return data_row.label_ids


def get_multi_label_ids_from_cache(cache):
    return np.array(
        [get_label_ids_from_data_row(data_row=datum["data_row"]) for datum in cache.iter_all()]
    )


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


def get_multiple_choice_label_ids_from_examples(task, examples):
    return np.array([task.CHOICE_BIMAP.a[example.label] for example in examples])


def get_multiple_choice_label_id_from_data_row(data_row):
    return data_row.label_id


def get_multiple_choice_labels_from_cache(cache):
    return np.array(
        [
            get_multiple_choice_label_id_from_data_row(data_row=datum["data_row"])
            for datum in cache.iter_all()
        ]
    )


def mean(*args) -> float:
    return float(np.mean(args))


def write_metrics(results, output_path, verbose=True):
    results_to_write = {}
    if "loss" in results:
        results_to_write["loss"] = results["loss"]
    if "metrics" in results:
        results_to_write["metrics"] = results["metrics"].to_dict()
    assert results_to_write
    metrics_str = json.dumps(results_to_write, indent=2)
    if verbose:
        print(metrics_str)
    with open(output_path, "w") as f:
        f.write(metrics_str)
