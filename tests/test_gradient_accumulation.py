import tempfile

from pkg_resources import resource_filename
import unittest
from unittest import mock

from jiant.__main__ import check_configurations
from jiant.tasks import tasks
from jiant.trainer import build_trainer
from jiant.utils.config import params_from_file


class TestGradientAccumulation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wic = tasks.WiCTask(self.temp_dir, 100, "wic", tokenizer_name="MosesTokenizer")

    def test_steps_between_gradient_accumulations_must_be_defined(self):
        self.args = params_from_file(resource_filename("jiant", "config/defaults.conf"))
        del self.args.accumulation_steps
        self.assertRaises(AssertionError, check_configurations, self.args, [], [])

    def test_steps_between_gradient_accumulations_cannot_be_set_to_a_negative_number(self):
        self.args = params_from_file(
            resource_filename("jiant", "config/defaults.conf"), "accumulation_steps = -1"
        )
        self.assertRaises(AssertionError, check_configurations, self.args, [], [])

    def test_by_default_steps_between_gradient_accumulations_is_set_to_1(self):
        with mock.patch("jiant.models.MultiTaskModel") as MockModel:
            self.args = params_from_file(resource_filename("jiant", "config/defaults.conf"))
            self.args.cuda = -1
            self.args.run_dir = self.temp_dir
            self.args.exp_dir = self.temp_dir
            model = MockModel()
            _, train_params, _, _ = build_trainer(
                self.args,
                self.args.cuda,
                ["wic"],
                model,
                self.args.run_dir,
                self.wic.val_metric_decreases,
                phase="pretrain",
            )
            self.assertEqual(train_params["accumulation_steps"], 1)

    def test_steps_between_gradient_accumulations_can_be_overridden_and_set_greater_than_1(self):
        with mock.patch("jiant.models.MultiTaskModel") as MockModel:
            self.args = params_from_file(
                resource_filename("jiant", "config/defaults.conf"), "accumulation_steps = 10"
            )
            self.args.cuda = -1
            self.args.run_dir = self.temp_dir
            self.args.exp_dir = self.temp_dir
            model = MockModel()
            _, train_params, _, _ = build_trainer(
                self.args,
                self.args.cuda,
                ["wic"],
                model,
                self.args.run_dir,
                self.wic.val_metric_decreases,
                phase="pretrain",
            )
            self.assertEqual(train_params["accumulation_steps"], 10)
