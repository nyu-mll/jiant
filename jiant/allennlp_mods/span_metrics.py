import collections
import string
import re

from typing import List, Dict

from allennlp.training.metrics.metric import Metric


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    From official ReCoRD eval script """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """ Compute normalized token level F1
    From official ReCoRD eval script """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """ Compute normalized exact match
    From official ReCoRD eval script """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """ Compute max metric between prediction and each ground truth.
    From official ReCoRD eval script """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class GenericSpanMetric(Metric):
    def __init__(self):
        self._metric_total = 0
        self._count = 0

    def metric_func(self, prediction, ground_truth):
        raise NotImplementedError

    def __call__(self, pred_str_list: List[str], gold_str_list: List[str]):
        # Breaking API here. Should we make Metric more general?
        metric_ls = [
            self.metric_func(prediction=pred_str, ground_truth=gold_str)
            for pred_str, gold_str in zip(pred_str_list, gold_str_list)
        ]

        self._metric_total += sum(metric_ls)
        self._count += len(metric_ls)

    def get_metric(self, reset: bool = False):
        metric = self._metric_total / self._count
        if reset:
            self.reset()
        return metric

    def reset(self):
        self._metric_total = 0
        self._count = 0


class F1SpanMetric(GenericSpanMetric):
    def metric_func(self, prediction, ground_truth):
        return f1_score(prediction=prediction, ground_truth=ground_truth)


class ExactMatchSpanMetric(GenericSpanMetric):
    def metric_func(self, prediction, ground_truth):
        return exact_match_score(prediction=prediction, ground_truth=ground_truth)
