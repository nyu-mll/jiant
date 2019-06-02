import csv
import os
import shutil
import tempfile
import unittest
import src.tasks.tasks as tasks
import torch


class TestUpdateMetricsAccuracy(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_dataset.tsv")
        self.task = tasks.WiCTask("", 100, "winograd", tokenizer_name="MosesTokenizer")
        predictions = torch.Tensor(
            [
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
                [[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                [[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
            ]
        )
        perfect_predictions = torch.Tensor(
            [
                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            ]
        )
        true_labels = torch.Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
        self.task.update_metrics(predictions[0], true_labels[0], tagmask=None)
        self.task.update_metrics(predictions[1], true_labels[1], tagmask=None)
        self.task.update_metrics(predictions[2], true_labels[2], tagmask=None)
        self.task.update_metrics(predictions[3], true_labels[3], tagmask=None)
        self.imperfect_metrics = self.task.get_metrics(reset=True)
        self.task.update_metrics(perfect_predictions[0], true_labels[0], tagmask=None)
        self.task.update_metrics(perfect_predictions[1], true_labels[1], tagmask=None)
        self.task.update_metrics(perfect_predictions[2], true_labels[2], tagmask=None)
        self.task.update_metrics(perfect_predictions[3], true_labels[3], tagmask=None)
        self.perfect_metrics = self.task.get_metrics(reset=True)

    def test_accuracy(self):
        # match predictions and labels to be the same format as in src/models.py
        # Only measures accuracy
        assert self.task.val_metric.split("_")[1] in list(self.imperfect_metrics.keys())
        assert self.imperfect_metrics["accuracy"] == 5.0 / 16.0
        assert self.perfect_metrics["accuracy"] == 1.0

    def test_f1(self):
        assert "f1" in self.imperfect_metrics
        assert round(self.imperfect_metrics["f1"], 5) == 0.35294
        assert round(self.perfect_metrics["f1"], 1) == 1.0
