import csv
import os
import shutil
import tempfile
import unittest
import src.tasks.tasks as tasks
import torch
from src import evaluate
import os.path
from main import evaluate_and_write
import pandas as pd
from src.models import MultiTaskModel
from unittest import mock
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance, Token, vocabulary
from ..allennlp_mods.numeric_field import NumericField
from allennlp.data.fields import (
    LabelField,
    ListField,
    MetadataField,
    TextField,
)


def model_forward(task, batch, predict=True):
    if task.name == "sts-b":
        return {"n_exs": 4, "preds": [5.0, 4.0]}
    return {"n_exs": 4, "preds": [0, 1, 1, 1]}


class TestWritePreds(unittest.TestCase):
    def sentence_to_text_field(self, sent, indexers):
        """ Helper function to map a sequence of tokens into a sequence of
        AllenNLP Tokens, then wrap in a TextField with the given indexers """
        return TextField(list(map(Token, sent)), token_indexers=indexers)

    def setUp(self):
        """
        Since we're testing write_preds, we need to mock model predictions and the parts 
        of the model, arguments, and trainer needed to write to predictions. 
        Unlike in update_metrics tests, the actual contents of the examples in val_data 
        is not the most important as long as it adheres to the API necessary for examples
        of that task. 
        """
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.temp_dir = self.current_path + "/tmp"
        if os.path.exists(self.temp_dir) is False:
            os.makedirs(self.temp_dir, exist_ok=True)

        # the current one 
        self.stsb = tasks.STSBTask(self.temp_dir, 100, "sts-b", tokenizer_name="MosesTokenizer")
        self.wic = tasks.WiCTask(self.temp_dir, 100, "wic", tokenizer_name="MosesTokenizer")
        stsb_val_preds = pd.DataFrame(
            data=[{
                "idx": 0,
                "labels": 5.00,
                "preds": 5.00,
                "sent1_str": "A man with a hard hat is dancing.",
                "sent2_str": "A man wearing a hard hat is dancing",
            }, 
            {
                "idx": 0,
                "labels": 4.750,
                "preds": 0.34,
                "sent1_str": "A young child is riding a horse.",
                "sent2_str": "A child is riding a horse.",
            }]
        )
        wic_val_preds = pd.DataFrame(
            data=[{
                "idx": 0,
                "sent1": "Room and board. ",
                "sent2": "He nailed boards across the windows.",
                "labels": 0,
            }, 
            {
                "idx": 0,
                "sent1": "Hook a fish",
                "sent2": "He hooked a snake accidentally , and was so scared he dropped his rod into the water .",
                "labels": 1,
            }]

        )
        indexers = {"bert_wpm_pretokenized": SingleIdTokenIndexer("bert-xe-cased")}
        self.wic.val_data = [
            Instance(
                {
                    "sent1_str": MetadataField("Room and board yo."),
                    "sent2_str": MetadataField("He nailed boards"),
                    "idx": LabelField(1, skip_indexing=True),
                    "idx2": NumericField(2),
                    "idx1": NumericField(3),
                    "inputs": self.sentence_to_text_field(
                        ["Whats", "up", "right", "now"], indexers
                    ),
                    "labels": LabelField(0, skip_indexing=1),
                }
            ),
            Instance(
                {
                    "sent1_str": MetadataField("C ##ir ##culate a rumor ."),
                    "sent2_str": MetadataField(
                        "This letter is being circulated among the faculty ."
                    ),
                    "idx": LabelField(1, skip_indexing=True),
                    "idx2": NumericField(2),
                    "idx1": NumericField(3),
                    "inputs": self.sentence_to_text_field(
                        ["Whats", "up", "right", "now"], indexers
                    ),
                    "labels": LabelField(0, skip_indexing=1),
                }
            ),
            Instance(
                {
                    "sent1_str": MetadataField("Hook a fish'"),
                    "sent2_str": MetadataField(
                        "He hooked a snake accidentally , and was so scared he dropped his rod into the water ."
                    ),
                    "idx": LabelField(1, skip_indexing=True),
                    "idx2": NumericField(2),
                    "idx1": NumericField(3),
                    "inputs": self.sentence_to_text_field(
                        ["Whats", "up", "right", "now"], indexers
                    ),
                    "labels": LabelField(1, skip_indexing=1),
                }
            ),
            Instance(
                {
                    "sent1_str": MetadataField(
                        "For recreation he wrote poetry and solved cross ##word puzzles ."
                    ),
                    "sent2_str": MetadataField(
                        "Drug abuse is often regarded as a form of recreation ."
                    ),
                    "idx": LabelField(1, skip_indexing=True),
                    "idx2": NumericField(2),
                    "idx1": NumericField(3),
                    "inputs": self.sentence_to_text_field(
                        ["Whats", "up", "right", "now"], indexers
                    ),
                    "labels": LabelField(1, skip_indexing=1),
                }
            ),
        ]
        self.val_preds = {"sts-b": stsb_val_preds, "wic": wic_val_preds}
        self.vocab = vocabulary.Vocabulary.from_instances(self.wic.val_data)
        self.vocab.add_token_to_namespace("True", "wic_tags")
        for data in self.wic.val_data:
            data.index_fields(self.vocab)
        self.glue_tasks = [self.stsb, self.wic]
        self.args = mock.Mock()
        self.args.batch_size = 4
        self.args.cuda = -1
        self.args.run_dir = self.temp_dir
        self.args.exp_dir = ""
    def test_if_path_exists(self):
        assert os.path.exists(self.temp_dir)

    def test_write_preds_does_run(self):
        evaluate.write_preds(
            self.glue_tasks, self.val_preds, self.temp_dir, "test", strict_glue_format=True
        )
        assert (
            os.path.exists(self.temp_dir + "/STS-B.tsv")
            and os.path.exists(self.temp_dir + "/WiC.jsonl")
        )

    def test_write_preds_glue(self):
        evaluate.write_preds(
            self.glue_tasks, self.val_preds, self.temp_dir, "test", strict_glue_format=True
        )
        stsb_predictions = pd.read_csv(self.temp_dir + "/STS-B.tsv", sep="\t")
        assert "index" in stsb_predictions.columns and "prediction" in stsb_predictions.columns
        assert stsb_predictions.iloc[0]["prediction"] == 5.00
        assert stsb_predictions.iloc[1]["prediction"] == 1.7
        
    def test_write_preds_superglue(self):
        """
        Ensure that SuperGLUE write predictions for test is saved to the correct file 
        format.
        """
        evaluate.write_preds(
            self.glue_tasks, self.val_preds, self.temp_dir, "test", strict_glue_format=True
        )
        wic_predictions = pd.read_json(self.temp_dir + "/WiC.jsonl", lines=True)
        assert "idx" in wic_predictions.columns and "label" in wic_predictions.columns
        assert wic_predictions.iloc[0]["label"] == "false"
        assert wic_predictions.iloc[1]["label"] == "true"

    @mock.patch("src.models.MultiTaskModel.forward", side_effect=model_forward)
    def test_evaluate_and_write_does_run(self, model_forward_function):
        """
        Testing that evaluate_and_write runs without breaking.
        """
        with mock.patch("src.models.MultiTaskModel") as MockModel:
            MockModel.return_value.eval.return_value = None
            MockModel.return_value.forward = model_forward
            MockModel.use_bert = 1
            model = MockModel()
            evaluate_and_write(self.args, model, [self.wic], splits_to_write="val")

    def tear_down(self):
        shutil.rmtree(self.temp_dir)
