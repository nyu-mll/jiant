import csv
import os
import os.path
import shutil
import tempfile
import unittest
from unittest import mock
import torch
import pandas as pd

from jiant import evaluate
import jiant.tasks.tasks as tasks
from jiant.models import MultiTaskModel
from jiant.__main__ import evaluate_and_write

from jiant.allennlp_mods.numeric_field import NumericField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance, Token, vocabulary
from allennlp.data.fields import LabelField, ListField, MetadataField, TextField


def model_forward(task, batch, predict=True):
    if task.name == "sts-b":
        logits = torch.Tensor([0.6, 0.4])
        labels = torch.Tensor([0.875, 0.6])
        out = {"logits": logits, "labels": labels, "n_exs": 2, "preds": [1.0, 0.8]}
    elif task.name == "wic":
        logits = torch.Tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        labels = torch.LongTensor([0, 1, 1, 0])
        out = {"logits": logits, "labels": labels, "n_exs": 4, "preds": [0, 1, 1, 1]}
    else:
        raise ValueError("Unexpected task found")

    task.update_metrics(out, batch)
    return out


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
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_dataset.tsv")
        self.stsb = tasks.STSBTask(self.temp_dir, 100, "sts-b", tokenizer_name="MosesTokenizer")
        self.wic = tasks.WiCTask(self.temp_dir, 100, "wic", tokenizer_name="MosesTokenizer")
        stsb_val_preds = pd.DataFrame(
            data=[
                {
                    "idx": 0,
                    "labels": 1.00,
                    "preds": 1.00,
                    "sent1_str": "A man with a hard hat is dancing.",
                    "sent2_str": "A man wearing a hard hat is dancing",
                },
                {
                    "idx": 1,
                    "labels": 0.950,
                    "preds": 0.34,
                    "sent1_str": "A young child is riding a horse.",
                    "sent2_str": "A child is riding a horse.",
                },
            ]
        )
        wic_val_preds = pd.DataFrame(
            data=[
                {
                    "idx": 0,
                    "sent1": "Room and board. ",
                    "sent2": "He nailed boards across the windows.",
                    "labels": 0,
                    "preds": 0,
                },
                {
                    "idx": 1,
                    "sent1": "Hook a fish",
                    "sent2": "He hooked a snake accidentally.",
                    "labels": 1,
                    "preds": 1,
                },
            ]
        )
        indexers = {"bert_cased": SingleIdTokenIndexer("bert-xe-cased")}
        self.wic.set_instance_iterable(
            "val",
            [
                Instance(
                    {
                        "sent1_str": MetadataField("Room and board."),
                        "sent2_str": MetadataField("He nailed boards"),
                        "idx": LabelField(0, skip_indexing=True),
                        "idx2": NumericField(2),
                        "idx1": NumericField(3),
                        "inputs": self.sentence_to_text_field(
                            [
                                "[CLS]",
                                "Room",
                                "and",
                                "Board",
                                ".",
                                "[SEP]",
                                "He",
                                "nailed",
                                "boards",
                                "[SEP]",
                            ],
                            indexers,
                        ),
                        "labels": LabelField(0, skip_indexing=1),
                    }
                ),
                Instance(
                    {
                        "sent1_str": MetadataField("C ##ir ##culate a rumor ."),
                        "sent2_str": MetadataField("This letter is being circulated"),
                        "idx": LabelField(1, skip_indexing=True),
                        "idx2": NumericField(2),
                        "idx1": NumericField(3),
                        "inputs": self.sentence_to_text_field(
                            [
                                "[CLS]",
                                "C",
                                "##ir",
                                "##culate",
                                "a",
                                "rumor",
                                "[SEP]",
                                "This",
                                "##let",
                                "##ter",
                                "is",
                                "being",
                                "c",
                                "##ir",
                                "##culated",
                                "[SEP]",
                            ],
                            indexers,
                        ),
                        "labels": LabelField(0, skip_indexing=1),
                    }
                ),
                Instance(
                    {
                        "sent1_str": MetadataField("Hook a fish'"),
                        "sent2_str": MetadataField("He hooked a snake accidentally"),
                        "idx": LabelField(2, skip_indexing=True),
                        "idx2": NumericField(2),
                        "idx1": NumericField(3),
                        "inputs": self.sentence_to_text_field(
                            [
                                "[CLS]",
                                "Hook",
                                "a",
                                "fish",
                                "[SEP]",
                                "He",
                                "hooked",
                                "a",
                                "snake",
                                "accidentally",
                                "[SEP]",
                            ],
                            indexers,
                        ),
                        "labels": LabelField(1, skip_indexing=1),
                    }
                ),
                Instance(
                    {
                        "sent1_str": MetadataField("For recreation he wrote poetry."),
                        "sent2_str": MetadataField("Drug abuse is often regarded as recreation ."),
                        "idx": LabelField(3, skip_indexing=True),
                        "idx2": NumericField(2),
                        "idx1": NumericField(3),
                        "inputs": self.sentence_to_text_field(
                            [
                                "[CLS]",
                                "For",
                                "re",
                                "##creation",
                                "he",
                                "wrote",
                                "poetry",
                                "[SEP]",
                                "Drug",
                                "abuse",
                                "is",
                                "often",
                                "re",
                                "##garded",
                                "as",
                                "re",
                                "##creation",
                                "[SEP]",
                            ],
                            indexers,
                        ),
                        "labels": LabelField(1, skip_indexing=1),
                    }
                ),
            ],
        )
        self.val_preds = {"sts-b": stsb_val_preds, "wic": wic_val_preds}
        self.vocab = vocabulary.Vocabulary.from_instances(self.wic.get_instance_iterable("val"))
        self.vocab.add_token_to_namespace("True", "wic_tags")
        for data in self.wic.get_instance_iterable("val"):
            data.index_fields(self.vocab)
        self.glue_tasks = [self.stsb, self.wic]
        self.args = mock.Mock()
        self.args.batch_size = 4
        self.args.cuda = -1
        self.args.run_dir = self.temp_dir
        self.args.exp_dir = ""

    def test_write_preds_does_run(self):
        evaluate.write_preds(
            self.glue_tasks, self.val_preds, self.temp_dir, "test", strict_glue_format=True
        )
        assert os.path.exists(self.temp_dir + "/STS-B.tsv") and os.path.exists(
            self.temp_dir + "/WiC.jsonl"
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
            [self.wic], self.val_preds, self.temp_dir, "test", strict_glue_format=True
        )
        wic_predictions = pd.read_json(self.temp_dir + "/WiC.jsonl", lines=True)
        assert "idx" in wic_predictions.columns and "label" in wic_predictions.columns
        assert wic_predictions.iloc[0]["label"] == "false"
        assert wic_predictions.iloc[1]["label"] == "true"

    @mock.patch("jiant.models.MultiTaskModel.forward", side_effect=model_forward)
    def test_evaluate_and_write_does_run(self, model_forward_function):
        """
        Testing that evaluate_and_write runs without breaking.
        """
        with mock.patch("jiant.models.MultiTaskModel") as MockModel:
            MockModel.return_value.eval.return_value = None
            MockModel.return_value.forward = model_forward
            MockModel.use_bert = 1
            model = MockModel()
            evaluate_and_write(self.args, model, [self.wic], splits_to_write="val", cuda_device=-1)

    def tear_down(self):
        shutil.rmtree(self.temp_dir)
