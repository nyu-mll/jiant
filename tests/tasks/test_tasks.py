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

    def test_tasks_get_train_instance_generators_without_phase(self):
        task = Task(name="dummy_name", tokenizer_name="dummy_tokenizer_name")
        train_iterable_instance_generator = [1, 2, 3]
        task.set_instance_generator("train", train_iterable_instance_generator, "target_train")
        self.assertRaises(ValueError, task.get_instance_generator, "train")

    def test_tasks_set_and_get_instance_generators(self):
        task = Task(name="dummy_name", tokenizer_name="dummy_tokenizer_name")
        val_iterable_instance_generator = [1, 2, 3]
        test_iterable_instance_generator = [4, 5, 6]
        train_pretrain_iterable_instance_generator = [7, 8]
        train_target_train_iterable_instance_generator = [9]
        task.set_instance_generator("val", val_iterable_instance_generator)
        task.set_instance_generator("test", test_iterable_instance_generator)
        task.set_instance_generator("train", train_pretrain_iterable_instance_generator, "pretrain")
        task.set_instance_generator(
            "train", train_target_train_iterable_instance_generator, "target_train"
        )
        retreived_val_iterable_instance_generator = task.get_instance_generator("val")
        retreived_test_iterable_instance_generator = task.get_instance_generator("test")
        retreived_train_pretrain_iterable_instance_generator = task.get_instance_generator(
            "train", "pretrain"
        )
        retreived_train_target_iterable_instance_generator = task.get_instance_generator(
            "train", "target_train"
        )
        self.assertListEqual(
            val_iterable_instance_generator, retreived_val_iterable_instance_generator
        )
        self.assertListEqual(
            test_iterable_instance_generator, retreived_test_iterable_instance_generator
        )
        self.assertListEqual(
            train_pretrain_iterable_instance_generator,
            retreived_train_pretrain_iterable_instance_generator,
        )
        self.assertListEqual(
            train_target_train_iterable_instance_generator,
            retreived_train_target_iterable_instance_generator,
        )
