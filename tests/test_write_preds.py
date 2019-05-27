
import csv
import os
import shutil
import tempfile
import unittest
import src.tasks.tasks as tasks
import torch
import src.evaluate as evaluate
import os.path
from main import evaluate_and_write
import pandas as pd
from unittest.mock import MagicMock
from src.model import MultiTaskModel

class TestWritePreds(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(temp_dir, "temp_dataset.tsv")
        stsb = tasks.STSBTask(temp_dir, 100, "sts-b", tokenizer_name="MosesTokenizer")
        mrpc = tasks.MRPCTask(temp_dir, 100, "mrpc", tokenizer_name="MosesTokenizer")
        wic = tasks.WiCTask(temp_dir, 100, "wic", tokenizer_name="MosesTokenizer")
        stsb_val_preds = pd.DataFrame(data={"idx": [0], "labels":[0], "preds":[0], "sent1_str":["When Tommy dropped his ice cream , Timmy"], "sent2_str":["Father gave Timmy a stern look" ]})
        mrpc_val_preds = pd.DataFrame(data={"idx": [0], "labels": [1], "preds": [1], "sentence_1": [" A man with a hard hat is dancing "], "sentence_2": [" A man wearing a hard hat is dancing"]})
        wic_val_preds = pd.DataFrame(data={"idx": [0], "sent1": ["Room and board. "], "sent2": ["He nailed boards across the windows."], "labels": [1]})
        val_preds = {"sts-b":stsb_val_preds, "mrpc": mrpc_val_preds, "wic": wic_val_preds}
        self.glue_tasks = [stsb, mrpc, wic]
        self.model = MultiTaskModel()
        self.model.forward = MagicMock(return_value=[0, 1])
        self.model.test_dataset = []

    def test_write_preds_does_run(self):
        evaluate.write_preds(glue_tasks, val_preds, temp_dir, "test", strict_glue_format=True)
        assert os.path.exists(temp_dir+"/STS-B.tsv") and os.path.exists(temp_dir + "/MRPC.tsv") and os.path.exists(temp_dir + "/WIC.jsonl")

    def test_write_preds_glue(self):
        stsb_predictions = pd.read_csv(temp_dir + "/STS-B.tsv", sep="\t")
        assert "index" in stsb_predictions.columns and "prediction" in stsb_predictions.columns
        assert stsb_predictions.iloc[0]["label"] = 0
        mrpc_predictions = pd.read_json(temp_dir + "/WIC.jsonl", lines=True)
        assert "idx" in mrpc_predictions.columns and "label" in mrpc_predictions.columns
        assert mrpc_predictions.iloc[0]["label"] = True

    def test_evaluate_and_write(self):
        args = {"write_strict_glue_format": True, "cuda": -1, \
                "run_dir": temp_dir, "exp_dir": "", \
                "run_name": "test", "batch_size": 4}
        evaluate_and_write(args, model, self.tasks, splits_to_write="val,test")
        # test that the resulting results is correct.


