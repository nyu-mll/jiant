import unittest
from src.trainer import build_trainer_params
from src.utils.config import params_from_file


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
        DEFAULTS_PATH = "config/defaults.conf"  # To get other required values.
        params = params_from_file(DEFAULTS_PATH, HOCON)
        self.processed_pretrain_params = build_trainer_params(
            params, ["mnli", "qqp"], phase="pretrain"
        )
        self.processed_mnli_target_params = build_trainer_params(
            params, ["mnli"], phase="target_train"
        )
        self.processed_qqp_target_params = build_trainer_params(
            params, ["qqp"], phase="target_train"
        )

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
