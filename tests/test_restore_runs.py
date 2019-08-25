import os
import os.path
import shutil
import tempfile
import unittest
from unittest import mock

from jiant import evaluate
import jiant.tasks.tasks as tasks
from jiant.utils import utils
from jiant.__main__ import evaluate_and_write, get_best_checkpoint_path


class TestRestoreRuns(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mrpc = tasks.MRPCTask(self.temp_dir, 100, "mrpc", tokenizer_name="MosesTokenizer")
        self.sst = tasks.MRPCTask(self.temp_dir, 100, "sst", tokenizer_name="MosesTokenizer")
        os.mkdir(os.path.join(self.temp_dir, "mrpc"))
        os.mkdir(os.path.join(self.temp_dir, "sst"))
        for type_name in ["model", "task", "training", "metric"]:
            open(
                os.path.join(self.temp_dir, "{}_state_pretrain_val_1.th".format(type_name)), "w"
            ).close()
            open(
                os.path.join(self.temp_dir, "{}_state_pretrain_val_2.th".format(type_name)), "w"
            ).close()
            open(
                os.path.join(self.temp_dir, "{}_state_pretrain_val_3.best.th".format(type_name)),
                "w",
            ).close()
            open(
                os.path.join(
                    self.temp_dir, "mrpc", "{}_state_target_train_val_1.best.th".format(type_name)
                ),
                "w",
            ).close()
            open(
                os.path.join(
                    self.temp_dir, "mrpc", "{}_state_target_train_val_2.th".format(type_name)
                ),
                "w",
            ).close()

    def test_check_for_previous_checkpoints(self):
        # Testing that check_for_previous_checkpoints returns the correct checkpoints given
        # the state of a run directory.
        tasks = [self.mrpc, self.sst]
        task_directory, max_epoch, suffix = utils.check_for_previous_checkpoints(
            self.temp_dir, tasks, phase="pretrain", load_model=True
        )
        assert task_directory == "" and max_epoch == 3 and suffix == "state_pretrain_val_3.best.th"
        task_directory, max_epoch, suffix = utils.check_for_previous_checkpoints(
            self.temp_dir, tasks, phase="target_train", load_model=True
        )
        assert (
            task_directory == "mrpc" and max_epoch == 2 and suffix == "state_target_train_val_2.th"
        )
        # Test partial checkpoints.
        # If <4 checkpoints are found for an epoch, we do not count that epoch as
        # the most recent.
        for type_name in ["model", "task"]:
            open(
                os.path.join(
                    self.temp_dir, "sst", "{}_state_target_train_val_1.best.th".format(type_name)
                ),
                "w",
            ).close()
        task_directory, max_epoch, suffix = utils.check_for_previous_checkpoints(
            self.temp_dir, tasks, phase="target_train", load_model=True
        )
        # Even though there are partial checkpoints in the sst directory,
        # it still will return the most recent checkpoint as in mrpc.
        assert (
            task_directory == "mrpc" and max_epoch == 2 and suffix == "state_target_train_val_2.th"
        )
        for type_name in ["training", "metric"]:
            open(
                os.path.join(
                    self.temp_dir, "sst", "{}_state_target_train_val_1.best.th".format(type_name)
                ),
                "w",
            ).close()
            open(
                os.path.join(
                    self.temp_dir, "sst", "{}_state_target_train_val_2.best.th".format(type_name)
                ),
                "w",
            ).close()
        task_directory, max_epoch, suffix = utils.check_for_previous_checkpoints(
            self.temp_dir, tasks, phase="target_train", load_model=True
        )
        # Now that there is a complete set of 4 checkpoints in the sst directory,
        # the function should return sst as the directory with the most recent
        # checkpoint.
        assert (
            task_directory == "sst"
            and max_epoch == 1
            and suffix == "state_target_train_val_1.best.th"
        )

    def test_check_for_previous_ckpt_assert(self):
        # Testing that if args.load_model=0 and there are checkpoints in pretrain or target_train,
        # check_for_previous_checkpoints throws an error.
        with self.assertRaises(AssertionError) as error:
            utils.check_for_previous_checkpoints(
                self.temp_dir, tasks, phase="pretrain", load_model=False
            )
            utils.check_for_previous_checkpoints(
                self.temp_dir, tasks, phase="target_train", load_model=False
            )

    def test_find_last_checkpoint_epoch(self):
        # Testing path-finding logic of find_last_checkpoint_epoch function.
        max_epoch, suffix = utils.find_last_checkpoint_epoch(
            self.temp_dir, search_phase="pretrain", task_name=""
        )
        assert max_epoch == 3 and suffix == "state_pretrain_val_3.best.th"
        max_epoch, suffix = utils.find_last_checkpoint_epoch(
            self.temp_dir, search_phase="target_train", task_name="sst"
        )
        assert max_epoch == -1 and suffix is None
        max_epoch, suffix = utils.find_last_checkpoint_epoch(
            self.temp_dir, search_phase="target_train", task_name="mrpc"
        )
        assert max_epoch == 2 and suffix == "state_target_train_val_2.th"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
