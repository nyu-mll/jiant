import unittest
import torch

from jiant.metrics.nli_metrics import NLITwoClassAccuracy


class TestNLIMetric(unittest.TestCase):
    def test_two_class_acc_w_two_class_data_and_model(self):
        nli_scorer = NLITwoClassAccuracy()

        # Note: predictions are of shape num_batches x batch_size x num_classes
        predictions = torch.Tensor([[[0, 1], [0, 1]], [[1, 0], [1, 0]]])
        true_labels = torch.Tensor([[1, 1], [0, 0]])
        nli_scorer(predictions, true_labels)
        acc = nli_scorer.get_metric(reset=True)
        assert acc == 1.0

        predictions = torch.Tensor([[[1, 0], [1, 0]], [[1, 0], [0, 1]]])
        true_labels = torch.Tensor([[1, 1], [0, 0]])
        nli_scorer(predictions, true_labels)
        acc = nli_scorer.get_metric(reset=True)
        assert acc == 1.0 / 4.0

    def test_two_class_acc_w_two_class_data(self):
        nli_scorer = NLITwoClassAccuracy()

        # Note: predictions are of shape num_batches x batch_size x num_classes
        predictions = torch.Tensor([[[0, 1, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]])
        true_labels = torch.Tensor([[1, 1], [0, 0]])
        nli_scorer(predictions, true_labels)
        acc = nli_scorer.get_metric(reset=True)
        assert acc == 1.0

        predictions = torch.Tensor([[[1, 0, 0], [1, 0, 0]], [[0, 0, 1], [0, 1, 0]]])
        true_labels = torch.Tensor([[1, 1], [0, 0]])
        nli_scorer(predictions, true_labels)
        acc = nli_scorer.get_metric(reset=True)
        assert acc == 1.0 / 4.0

    def test_two_class_acc_w_two_class_model(self):
        nli_scorer = NLITwoClassAccuracy()

        # Note: predictions are of shape num_batches x batch_size x num_classes
        predictions = torch.Tensor([[[0, 1], [0, 1]], [[1, 0], [1, 0]]])
        true_labels = torch.Tensor([[1, 1], [2, 0]])
        nli_scorer(predictions, true_labels)
        acc = nli_scorer.get_metric(reset=True)
        assert acc == 1.0

        predictions = torch.Tensor([[[1, 0], [1, 0]], [[1, 0], [0, 1]]])
        true_labels = torch.Tensor([[1, 1], [2, 0]])
        nli_scorer(predictions, true_labels)
        acc = nli_scorer.get_metric(reset=True)
        assert acc == 1.0 / 4.0
