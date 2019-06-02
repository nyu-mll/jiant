
import csv
import os
import shutil
import tempfile
import unittest
import src.trainer as trainer
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


def build_trainer_params(args, task_names, phase="pretrain"):
    return {"lr": 1e-05, "warmup": 4000, "d_hid": 1024, "scheduler_threshold": 0.0001, "lr_patience", 4, "lr_decay_factor": 0.05, "max_grad_norm": 5.0, "val_interval": 500, "keep_all_checkpoints":0, "max_epochs": 2, "scheduler_threshold": 0.05, "patience": 4, "dec_val_scale": 4, "training_data_fraction": 1, "val_interval": 1, "cuda": -1, "keep_all_checkpoints": 1}


class TestCheckpionting(unittest.TestCase):
    def sentence_to_text_field(self, sent, indexers):
        """ Helper function to map a sequence of tokens into a sequence of
        AllenNLP Tokens, then wrap in a TextField with the given indexers """
        return TextField(list(map(Token, sent)), token_indexers=indexers)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_dataset.tsv")
        self.stsb = tasks.STSBTask(self.temp_dir, 100, "sts-b", tokenizer_name="MosesTokenizer")
        self.mrpc = tasks.MRPCTask(self.temp_dir, 100, "mrpc", tokenizer_name="MosesTokenizer")
        self.wic = tasks.WiCTask(self.temp_dir, 100, "wic", tokenizer_name="MosesTokenizer")
        stsb_val_preds = pd.DataFrame(
            data={
                "idx": [0],
                "labels": [0],
                "preds": [0],
                "sent1_str": ["When Tommy dropped his ice cream , Timmy"],
                "sent2_str": ["Father gave Timmy a stern look"],
            }
        )
        mrpc_val_preds = pd.DataFrame(
            data={
                "idx": [0],
                "labels": [1],
                "preds": [1],
                "sentence_1": [" A man with a hard hat is dancing "],
                "sentence_2": [" A man wearing a hard hat is dancing"],
            }
        )
        wic_val_preds = pd.DataFrame(
            data={
                "idx": [0],
                "sent1": ["Room and board. "],
                "sent2": ["He nailed boards across the windows."],
                "labels": [1],
            }
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
        self.val_preds = {"sts-b": stsb_val_preds, "mrpc": mrpc_val_preds, "wic": wic_val_preds}
        self.vocab = vocabulary.Vocabulary.from_instances(self.wic.val_data)
        self.vocab.add_token_to_namespace("True", "wic_tags")
        for data in self.wic.val_data:
            data.index_fields(self.vocab)
        self.glue_tasks = [self.stsb, self.mrpc, self.wic]
        self.args = mock.Mock()
        self.args.batch_size = 4
        self.args.cuda = -1
        self.args.run_dir = self.temp_dir
        self.args.exp_dir = ""

    @mock.patch("src.trainer.build_trainer_params", side_effect=build_trainer_params)
    def test_checkpointing_does_run(self, build_trainer_params_function):
    	# test trainer.py, and run for pretrain and make sure it saves for two epochs.
    	test_trainer, _, opt_params, schd_params = trainer.build_trainer(
                self.args,
                ["wic"],
                model,
                self.args.run_dir,
                self.wic.val_metric_decreases,
                phase="target_train",
            )
    	test_trainer._SamplingMultiTaskTrainer__save_checkpoint(
                        {"pass": 10, "epoch": 1, "should_stop": 0},
                        phase="target_train", 
                        new_best_macro=True,
                    )
    	test_trainer._SamplingMultiTaskTrainer__save_checkpoint(
                        {"pass": 10, "epoch": 1, "should_stop": 0},
                        phase="target_train", 
                        new_best_macro=False,
                    )
        # check for target task training. 
        assert (
            os.path.exists(self.temp_dir + "/model_state_target_train_epoch_1.best_macro.th")
            and os.path.exists(self.temp_dir + "/model_state_pretrain_epoch_10.th ")
        )

    def test_reload_checkpointing(self):
    	# make sure that it picks up from checkpointing adn that other htan random variation, ther eis 
    	# deterministic. 
    	pass 
    	
    def test_evaluate_model_checkpoint(self):
    	# evaluate best, not most recent 
    	pass

    def test_transfer_checkpiont(self):
    	# we should load the best checkpoint for that model, not the most recent
    	pass

    def test_finetuning_checkpoint_for_multiple_tasks(self):
    	# make sure that we evaluate from the same pretrain_tasks checkpoint when finetuning
    	pass
