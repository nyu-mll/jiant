import numpy as np
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("spanf1")
class SpanF1Measure(Metric):
    def __init__(self):
        self._f1_total = 0
        self._count = 0

    def __call__(
        self,
        pred_start: torch.Tensor,
        pred_end: torch.Tensor,
        gold_start: torch.Tensor,
        gold_end: torch.Tensor,
    ):
        pred_start = np.argmax(pred_start.detach().cpu().numpy(), -1)
        pred_end = np.argmax(pred_end.detach().cpu().numpy(), -1)
        pred_end = np.maximum(pred_end, pred_start + 1)  # make end at least start + 1
        gold_start = gold_start.cpu().numpy().reshape(-1)
        gold_end = gold_end.cpu().numpy().reshape(-1)

        num_same = np.maximum(
            np.minimum(pred_end, gold_end) - np.maximum(pred_start, gold_start), 0
        )
        precision = num_same / (pred_end - pred_start)
        recall = num_same / (gold_end - gold_start)
        f1 = 2 * precision * recall / (precision + recall)

        # F1 is 0 is there is no overlap
        f1[np.isnan(f1)] = 0

        self._f1_total += f1.sum(0)
        self._count += len(pred_start)

    def get_metric(self, reset: bool = False):
        f1 = self._f1_total / self._count
        if reset:
            self.reset()
        return f1

    def reset(self):
        self._f1_total = 0
        self._count = 0
