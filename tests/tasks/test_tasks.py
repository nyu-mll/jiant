import logging
import unittest
from unittest import mock
from mock import patch
import tempfile
from jiant.tasks.registry import REGISTRY
from jiant.preprocess import get_tasks
from jiant.tasks import REGISTRY


class Record:
    def __init__(self, task_src_path, max_seq_len, name, tokenizer_name):
        self.example_counts = {"train": 100, "val": 90, "val": 88}

    def load_data(self):
        return

    def count_examples(self):
        return 0


class TestTasks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.args = mock.Mock()
        self.args.batch_size = 4
        self.args.cuda = -1
        self.args.run_dir = self.temp_dir
        self.args.exp_dir = ""
        self.args.reload_tasks = 1
        self.args.data_dir = ""
        self.args.pretrain_tasks = "record"
        self.args.target_tasks = "record"
        self.args.batch_size = 4
        self.args.cuda = [0, 1]
        self.args.tokenizer = "roberta-large"
        self.args.max_seq_len = 99

    def test_instantiate_all_tasks(self):
        """
        All tasks should be able to be instantiated without needing to access actual data

        Test may change if task instantiation signature changes.
        """
        logger = logging.getLogger()
        logger.setLevel(level=logging.CRITICAL)
        for name, (cls, _, kw) in REGISTRY.items():
            cls(
                "dummy_path",
                max_seq_len=1,
                name="dummy_name",
                tokenizer_name="dummy_tokenizer_name",
                **kw,
            )

    def test_batch_size(self):
        with patch.dict(REGISTRY, {"record": (Record, "", {})}, clear=True):
            with self.assertRaises(AssertionError) as error:
                get_tasks(self.args)
