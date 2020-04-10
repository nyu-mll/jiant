from pkg_resources import resource_filename
import unittest
from unittest import mock
from jiant.trainer import build_trainer_params
from jiant.utils.config import params_from_file
from jiant.__main__ import get_pretrain_stop_metric
from jiant.tasks.registry import REGISTRY


class TestBuildTrainerParams(unittest.TestCase):
    def setUp(self):
        HOCON = """
            lr = 123.456
            pretrain_data_fraction = .123
            target_train_data_fraction = .1234
            mnli = {
                lr = 4.56,
                batch_size = 123
                max_epochs = 456
                training_data_fraction = .456
            }
            qqp = {
                max_epochs = 789
            }
        """
        DEFAULTS_PATH = resource_filename(
            "jiant", "config/defaults.conf"
        )  # To get other required values.
        params = params_from_file(DEFAULTS_PATH, HOCON)
        cuda_device = -1
        self.processed_pretrain_params = build_trainer_params(
            params, cuda_device, ["mnli", "qqp"], phase="pretrain"
        )
        self.processed_mnli_target_params = build_trainer_params(
            params, cuda_device, ["mnli"], phase="target_train"
        )
        self.processed_qqp_target_params = build_trainer_params(
            params, cuda_device, ["qqp"], phase="target_train"
        )
        self.pretrain_tasks = []
        pretrain_task_registry = {
            "sst": REGISTRY["sst"],
            "winograd-coreference": REGISTRY["winograd-coreference"],
            "commitbank": REGISTRY["commitbank"],
        }
        for name, (cls, _, kw) in pretrain_task_registry.items():
            task = cls(
                "dummy_path", max_seq_len=1, name=name, tokenizer_name="dummy_tokenizer_name", **kw
            )
            self.pretrain_tasks.append(task)

    def test_pretrain_task_specific(self):
        # Task specific trainer parameters shouldn't apply during pretraining.
        assert self.processed_pretrain_params["lr"] == 123.456
        assert self.processed_pretrain_params["training_data_fraction"] == 0.123
        assert self.processed_pretrain_params["keep_all_checkpoints"] == 0  # From defaults

    def test_target_task_specific(self):
        # Target task parameters should be task specific when possible, and draw on defaults
        # otherwise.
        assert self.processed_mnli_target_params["lr"] == 4.56
        assert self.processed_qqp_target_params["lr"] == 123.456
        assert self.processed_mnli_target_params["max_epochs"] == 456
        assert self.processed_qqp_target_params["max_epochs"] == 789
        assert self.processed_mnli_target_params["training_data_fraction"] == 0.456
        assert self.processed_qqp_target_params["training_data_fraction"] == 0.1234
        assert self.processed_mnli_target_params["keep_all_checkpoints"] == 0  # From defaults
        assert self.processed_qqp_target_params["keep_all_checkpoints"] == 0  # From defaults

    def test_pretrain_stop_metric(self):
        self.args = mock.Mock()
        self.args.early_stopping_method = "auto"
        # Make sure auto case works as expected.
        assert (
            get_pretrain_stop_metric(self.args.early_stopping_method, self.pretrain_tasks)
            == "macro_avg"
        )
        sst_only_pretrain = [self.pretrain_tasks[0]]
        assert (
            get_pretrain_stop_metric(self.args.early_stopping_method, sst_only_pretrain)
            == "sst_accuracy"
        )
        self.args.early_stopping_method = "winograd-coreference"
        assert (
            get_pretrain_stop_metric(self.args.early_stopping_method, self.pretrain_tasks)
            == "winograd-coreference_acc"
        )
        # Case where if we set early_stopping_method to a task that is not included in pretrain_tasks
        self.args.early_stopping_method = "sst_accuracy"
        pretrain_tasks_no_sst = self.pretrain_tasks[1:]
        with self.assertRaises(ValueError):
            get_pretrain_stop_metric(self.args.early_stopping_method, pretrain_tasks_no_sst)
