import unittest
import torch

from jiant.tasks.registry import REGISTRY


class TestMLM(unittest.TestCase):
    def test(self):
        """
        All tasks should be able to be instantiated without needing to access actual data

        Test may change if task instantiation signature changes.
        """
        cls, _, kw = REGISTRY["mlm"]
        mlm_task = cls(
            "dummy_path",
            max_seq_len=1,
            name="dummy_name",
            tokenizer_name="dummy_tokenizer_name",
            **kw,
        )
        # At this point, the inputs + labels are padded by 2 due to AllenNLP.
        # ALBERT vocabulary is 300002. Here we test the edge case where one of
        # the indices is 300002.
        # We don't test inputs because it's from transformers, and it's hard
        # to test whne there's random masking
        # Labels should be shifted back by 2 to match the output of the model
        # since we shift it forward from AllenNLP vocab.
        inputs = torch.LongTensor([[4, 100, 30002, 50, 20, 5], [4, 10, 320, 29, 33, 5]])
        labels = torch.LongTensor([[4, 100, 30002, 50, 20, 5], [4, 10, 320, 29, 33, 5]])
        input_key = "albert"
        tokenizer_name = "albert-xxlarge-v2"
        _unk_id = 1
        _pad_id = 0
        max_pos = None
        mask_idx = 222
        labels = mlm_task.mlm_correct_labels(labels, input_key, _unk_id, _pad_id, max_pos)
        assert sum(labels[0].numpy() == [2, 98, 30000, 48, 18, 3]) == 6
        assert sum(labels[1].numpy() == [2, 8, 318, 27, 31, 3]) == 6
