import logging
import unittest
from unittest import mock
from unittest.mock import patch
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
        self.temp_dir = tempfile.mkdtemp()
        args = mock.Mock()
        args.batch_size = 4
        args.cuda = -1
        args.run_dir = self.temp_dir
        args.exp_dir = ""
        args.reload_tasks = 1
        args.data_dir = ""
        args.pretrain_tasks = "record"
        args.target_tasks = "record"
        args.batch_size = 4
        args.cuda = [0, 1]
        args.tokenizer = "roberta-large"
        args.max_seq_len = 99
        with patch.dict(REGISTRY, {"record": (Record, "", {})}, clear=True):
            with self.assertRaises(AssertionError) as error:
                get_tasks(args)
