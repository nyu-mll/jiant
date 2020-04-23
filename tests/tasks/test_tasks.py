import logging
import unittest

from jiant.tasks import Task
from jiant.tasks.registry import REGISTRY


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

    def test_tasks_get_train_instance_iterable_without_phase(self):
        task = Task(name="dummy_name", tokenizer_name="dummy_tokenizer_name")
        train_iterable = [1, 2, 3]
        task.set_instance_iterable("train", train_iterable, "target_train")
        self.assertRaises(ValueError, task.get_instance_iterable, "train")

    def test_tasks_set_and_get_instance_iterables(self):
        task = Task(name="dummy_name", tokenizer_name="dummy_tokenizer_name")
        val_iterable = [1, 2, 3]
        test_iterable = [4, 5, 6]
        train_pretrain_iterable = [7, 8]
        train_target_train_iterable = [9]
        task.set_instance_iterable("val", val_iterable)
        task.set_instance_iterable("test", test_iterable)
        task.set_instance_iterable("train", train_pretrain_iterable, "pretrain")
        task.set_instance_iterable("train", train_target_train_iterable, "target_train")
        retreived_val_iterable = task.get_instance_iterable("val")
        retreived_test_iterable = task.get_instance_iterable("test")
        retreived_train_pretrain_iterable = task.get_instance_iterable("train", "pretrain")
        retreived_train_target_iterable = task.get_instance_iterable("train", "target_train")
        self.assertListEqual(val_iterable, retreived_val_iterable)
        self.assertListEqual(test_iterable, retreived_test_iterable)
        self.assertListEqual(train_pretrain_iterable, retreived_train_pretrain_iterable)
        self.assertListEqual(train_target_train_iterable, retreived_train_target_iterable)
