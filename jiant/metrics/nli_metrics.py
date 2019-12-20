from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("nli_two_class_accuracy")
class NLITwoClassAccuracy(Metric):
    """
    Metric that evaluates two-way NLI classifiers on three-way data or vice-versa.

    Thhis computes standard accuracy, but collapses 'neutral' and 'contradiction'
    into one label, assuming that 'entailment' is at index 1 (as in the jiant
    implementations of SNLI, MNLI and RTE).

    Based on allennlp.training.metrics.CategoricalAccuracy.
    """

    def __init__(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError(
                "gold_labels must have dimension == predictions.size() - 1 but "
                "found tensor of shape: {}".format(predictions.size())
            )

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()

        top_one_preds = predictions.max(-1)[1].unsqueeze(-1)

        # NLI-specific filtering
        top_one_preds = top_one_preds == 1
        gold_labels = gold_labels == 1

        # This is of shape (batch_size, ..., top_one_preds).
        correct = top_one_preds.eq(gold_labels.unsqueeze(-1)).float()

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
