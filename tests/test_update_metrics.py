import csv
import os
import shutil
import tempfile
import unittest
import jiant.tasks.tasks as tasks
import torch


class TestUpdateMetricsAccuracy(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_dataset.tsv")
        self.task = tasks.QQPTask("", 100, "qqp", tokenizer_name="MosesTokenizer")
        # Note: predictions are of shape num_batches x batch_size x num_classes
        # Here, QQP is binary thus num_classes = 2 and we make batch_size = 2.
        predictions = torch.Tensor([[[1, 0], [1, 0]], [[1, 0], [0, 1]]])
        perfect_predictions = torch.Tensor([[[0, 1], [0, 1]], [[1, 0], [1, 0]]])
        true_labels = torch.Tensor([[1, 1], [0, 0]])

        one_batch_predictions = torch.Tensor([[0], [1]])
        one_batch_true = torch.Tensor([[1], [0]])
        self.task.update_metrics(predictions[0], true_labels[0], tagmask=None)
        self.task.update_metrics(predictions[1], true_labels[1], tagmask=None)
        self.imperfect_metrics = self.task.get_metrics(reset=True)
        self.task.update_metrics(perfect_predictions[0], true_labels[0], tagmask=None)
        self.task.update_metrics(perfect_predictions[1], true_labels[1], tagmask=None)
        self.perfect_metrics = self.task.get_metrics(reset=True)

    def test_accuracy(self):
        # match predictions and labels to be the same format as in jiant/models.py
        # only measures accuracy
        assert "acc_f1" in list(self.imperfect_metrics.keys())
        assert self.imperfect_metrics["accuracy"] == 1.0 / 4.0
        assert self.perfect_metrics["accuracy"] == 1.0

    def test_f1(self):
        assert "f1" in self.imperfect_metrics
        assert round(self.imperfect_metrics["f1"], 1) == 0.0
        assert round(self.perfect_metrics["f1"], 1) == 1.0

    def tear_down(self):
        shutil.rmtree(self.temp_dir)
