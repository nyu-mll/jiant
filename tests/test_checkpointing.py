import csv
import os
import os.path
import shutil
import tempfile
import unittest
from unittest import mock
import torch
import pandas as pd
import glob

from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Instance, Token, vocabulary
from allennlp.data.fields import LabelField, ListField, MetadataField, TextField
from allennlp.common.params import Params
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.training.learning_rate_schedulers import (  # pylint: disable=import-error
    LearningRateScheduler,
)
from allennlp.common.params import Params
from allennlp.training.optimizers import Optimizer
from jiant.allennlp_mods.numeric_field import NumericField

import jiant.trainer as trainer
from jiant.models import MultiTaskModel
import jiant.tasks.tasks as tasks
from jiant.__main__ import get_best_checkpoint_path


def build_trainer_params(args, cuda_device, task_names, phase="pretrain"):
    return {
        "lr": 1e-05,
        "val_data_limit": 10,
        "d_hid": 1024,
        "min_lr": 0.000,
        "max_vals": 4,
        "sent_enc": "null",
        "scheduler_threshold": 0.0001,
        "optimizer": "bert_adam",
        "lr_patience": 4,
        "lr_decay_factor": 0.05,
        "max_grad_norm": 5.0,
        "val_interval": 500,
        "keep_all_checkpoints": 0,
        "max_epochs": 2,
        "scheduler_threshold": 0.05,
        "patience": 4,
        "dec_val_scale": 4,
        "training_data_fraction": 1,
        "val_interval": 1,
        "cuda": cuda_device,
        "keep_all_checkpoints": 1,
    }


class TestCheckpointing(unittest.TestCase):
    def sentence_to_text_field(self, sent, indexers):
        """ Helper function to map a sequence of tokens into a sequence of
        AllenNLP Tokens, then wrap in a TextField with the given indexers """
        return TextField(list(map(Token, sent)), token_indexers=indexers)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "temp_dataset.tsv")
        self.wic = tasks.WiCTask(self.temp_dir, 100, "wic", tokenizer_name="MosesTokenizer")
        indexers = {"bert_cased": SingleIdTokenIndexer("bert-xe-cased")}
        self.wic.val_data = [
            Instance(
                {
                    "sent1_str": MetadataField("Room and board yo."),
                    "sent2_str": MetadataField("He nailed boards"),
                    "idx": LabelField(1, skip_indexing=True),
                    "idx2": NumericField(2),
                    "idx1": NumericField(3),
                    "inputs": self.sentence_to_text_field(
                        ["[CLS]", "Room", "and", "board", "yo", "[SEP]", "He", "nailed", "boards"],
                        indexers,
                    ),
                    "labels": LabelField(0, skip_indexing=1),
                }
            )
        ]
        self.vocab = vocabulary.Vocabulary.from_instances(self.wic.val_data)
        self.vocab.add_token_to_namespace("True", "wic_tags")
        for data in self.wic.val_data:
            data.index_fields(self.vocab)
        self.args = mock.Mock()
        self.args.batch_size = 4
        self.args.cuda = -1
        self.args.run_dir = self.temp_dir
        self.args.exp_dir = ""

    @mock.patch("jiant.trainer.build_trainer_params", side_effect=build_trainer_params)
    def test_checkpointing_does_run(self, build_trainer_params_function):
        # Check that checkpointing does run and does sanity checks that at each step
        # it saves the most recent checkpoint as well as the best checkpoint
        # correctly for both pretrain and target_train stages.
        with mock.patch("jiant.models.MultiTaskModel") as MockModel:
            import torch
            import copy
            import time
            from allennlp.common.params import Params

            MockModel.return_value.eval.return_value = None
            MockModel.return_value.state_dict.return_value = {"model1": {"requires_grad": True}}
            pad_dict = self.wic.val_data[0].get_padding_lengths()
            sorting_keys = []
            for field in pad_dict:
                for pad_field in pad_dict[field]:
                    sorting_keys.append((field, pad_field))
            iterator = BucketIterator(
                sorting_keys=sorting_keys,
                max_instances_in_memory=10000,
                batch_size=4,
                biggest_batch_first=True,
            )
            opt_params = Params({"type": "adam", "lr": 1e-05})
            opt_params2 = copy.deepcopy(opt_params)
            scheduler_params = Params(
                {
                    "type": "reduce_on_plateau",
                    "factor": 0.05,
                    "mode": "max",
                    "patience": 4,
                    "threshold": 0.05,
                    "threshold_mode": "abs",
                    "verbose": True,
                }
            )
            train_params = [
                (
                    "_text_field_embedder.model.encoder.layer.9.output.dense.bias",
                    torch.Tensor([0.1, 0.3, 0.4, 0.8]),
                ),
                ("sent_encoder.layer.1", torch.Tensor([0.1, 0.3, 0.4, 0.8])),
                ("type", torch.Tensor([0.1])),
            ]
            scheduler = LearningRateScheduler.from_params(
                Optimizer.from_params(train_params, opt_params2), copy.deepcopy(scheduler_params)
            )
            optimizer = Optimizer.from_params(train_params, copy.deepcopy(opt_params))
            _task_infos = {
                "wic": {
                    "iterator": iterator(self.wic.val_data, num_epochs=1),
                    "n_tr_batches": 1,
                    "loss": 0.0,
                    "tr_generator": iterator(self.wic.val_data, num_epochs=1),
                    "total_batches_trained": 400,
                    "n_batches_since_val": 0,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "stopped": False,
                    "last_log": time.time(),
                }
            }
            _metric_infos = {
                metric: {"hist": [], "stopped": False, "best": (-1, {})}
                for metric in [self.wic.val_metric]
            }
            MockModel.return_value._setup_training.return_value = _task_infos, _metric_infos

            class MockParams:
                def __init__(self, requires_grad):
                    self.requires_grad = requires_grad

            MockModel.return_value.named_parameters.return_value = [("model1", MockParams(True))]
            MockModel.use_bert = 1
            model = MockModel()
            cuda_device = -1
            pt_trainer, _, _, _ = trainer.build_trainer(
                self.args,
                cuda_device,
                ["wic"],  # here, we use WIC twice to reduce the amount of boiler-plate code
                model,
                self.args.run_dir,
                self.wic.val_metric_decreases,
                phase="pretrain",
            )

            tt_trainer, _, _, _ = trainer.build_trainer(
                self.args,
                cuda_device,
                ["wic"],
                model,
                self.args.run_dir,
                self.wic.val_metric_decreases,
                phase="target_train",
            )
            os.mkdir(os.path.join(self.temp_dir, "wic"))

            tt_trainer.task_to_metric_mapping = {self.wic.val_metric: self.wic.name}
            pt_trainer._task_infos = _task_infos
            pt_trainer._metric_infos = _metric_infos
            pt_trainer._optimizer = optimizer
            pt_trainer._scheduler = scheduler
            pt_trainer._save_checkpoint(
                {"step": 10, "validation_pass": 1, "should_stop": 0},
                tasks=[self.wic],
                phase="pretrain",
                new_best=True,
            )
            pt_trainer._save_checkpoint(
                {"step": 10, "validation_pass": 2, "should_stop": 0},
                tasks=[self.wic],
                phase="pretrain",
                new_best=True,
            )
            tt_trainer._task_infos = _task_infos
            tt_trainer._metric_infos = _metric_infos
            tt_trainer._optimizer = optimizer
            tt_trainer._scheduler = scheduler

            tt_trainer._save_checkpoint(
                {"step": 10, "validation_pass": 1, "should_stop": 0},
                tasks=[self.wic],
                phase="target_train",
                new_best=True,
            )
            tt_trainer._save_checkpoint(
                {"step": 10, "validation_pass": 2, "should_stop": 0},
                tasks=[self.wic],
                phase="target_train",
                new_best=False,
            )
            assert (
                os.path.exists(
                    os.path.join(self.temp_dir, "wic", "model_state_target_train_val_1.best.th")
                )
                and os.path.exists(
                    os.path.join(self.temp_dir, "wic", "model_state_target_train_val_2.th")
                )
                and os.path.exists(
                    os.path.join(self.temp_dir, "model_state_pretrain_val_2.best.th")
                )
                and os.path.exists(os.path.join(self.temp_dir, "model_state_pretrain_val_1.th"))
            )

            # Assert only one checkpoint is created for pretrain stage.
            pretrain_best_checkpoints = glob.glob(
                os.path.join(self.temp_dir, "model_state_pretrain_val_*.best.th")
            )
            assert len(pretrain_best_checkpoints) == 1

    def test_get_best_checkpointing(self):
        """
        Testing the get_best_checkpointing function for path logic.
        """
        self.args.load_target_train_checkpoint = os.path.join(self.temp_dir, "target_checkpoint")
        self.args.load_eval_checkpoint = ""
        open(self.args.load_target_train_checkpoint, "wb").close()
        target_ckpt = get_best_checkpoint_path(self.args, phase="target_train", task_name=None)
        assert target_ckpt == self.args.load_target_train_checkpoint

        # Load from best pretrain checkpoint.
        self.args.load_target_train_checkpoint = ""
        os.mkdir(os.path.join(self.temp_dir, "wic"))
        best_pretrain_path = os.path.join(self.temp_dir, "model_state_pretrain_val_1.best.th")
        open(best_pretrain_path, "wb").close()
        target_ckpt = get_best_checkpoint_path(self.args, phase="target_train", task_name=None)
        assert target_ckpt == best_pretrain_path

        # Load from the best target train phase checkpoint.
        best_target_train_path = os.path.join(
            self.temp_dir, "wic", "model_state_target_train_val_1.best.th"
        )
        open(best_target_train_path, "wb").close()
        target_ckpt = get_best_checkpoint_path(self.args, phase="eval", task_name="wic")
        assert target_ckpt == best_target_train_path

        # Load from pre-existing eval checkpoint.
        self.args.load_eval_checkpoint = str(os.path.join(self.temp_dir, "eval_checkpoint"))
        open(self.args.load_eval_checkpoint, "wb").close()
        target_ckpt = get_best_checkpoint_path(self.args, phase="eval", task_name="wic")
        target_ckpt == self.args.load_eval_checkpoint

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
