import unittest
import torch
import random
import copy
import numpy as np
from unittest import mock
from jiant.tasks.registry import REGISTRY
from jiant.modules.sentence_encoder import SentenceEncoder
from jiant.huggingface_transformers_interface.modules import HuggingfaceTransformersEmbedderModule
from jiant.modules.simple_modules import NullPhraseLayer


class TestMLM(unittest.TestCase):
    def setUp(self):
        cls, _, kw = REGISTRY["wikipedia_corpus_mlm"]
        self.MLMTask = cls(
            "wikipedia_corpus_mlm",
            max_seq_len=10,
            name="wikipedia_corpus_mlm",
            tokenizer_name="roberta-large",
            **kw,
        )
        args = mock.Mock()
        args.input_module = "roberta-large"
        args.transformer_max_layer = 0
        args.transfer_paradigm = "finetune"
        args.transformers_output_mode = "none"

        args.exp_dir = ""
        vocab = mock.Mock()
        vocab._padding_token = "<pad>"
        vocab.get_token_index.side_effect = lambda x: 0
        self.embedder = HuggingfaceTransformersEmbedderModule(args)
        self.embedder._mask_id = 3
        self.embedder.model = mock.Mock()
        self.embedder.model.config = mock.Mock()
        self.embedder.model.config.hidden_size = 100
        self.embedder._pad_id = 0
        self.embedder.max_pos = None
        self.embedder._unk_id = 1
        phrase_layer = NullPhraseLayer(512)
        self.sent_encoder = SentenceEncoder(
            vocab, self.embedder, 1, phrase_layer, skip_embs=0, dropout=0.2, sep_embs_for_skip=1
        )

    def test_indexing(self):
        # Make sure that inputs and labels are consistent with each other
        # after the transforms for the indices that are unchanged by
        # the dynamic masking.
        mask_idx = self.embedder._mask_id
        tokenizer_name = "roberta-large"
        orig_inputs = torch.LongTensor(
            [[4, 100, 30, 459, 2340, 2309, 40, 230, 100, 30, 459, 2340, 2309, 40, 230, 5]]
        )
        orig_labels = torch.LongTensor(
            [[4, 100, 30, 459, 2340, 2309, 40, 230, 100, 30, 459, 2340, 2309, 40, 230, 5]]
        )
        inputs, labels, indices_replaced, _, masked_indices, labels_after_shift = self.MLMTask.mlm_dynamic_masking(
            orig_inputs, orig_labels, mask_idx, tokenizer_name, self.sent_encoder
        )
        inputs, _ = self.embedder.correct_sent_indexing({"roberta": inputs})
        inputs_unchanged = inputs[~masked_indices]
        labels_after_shift = labels_after_shift[~masked_indices]
        for input_unchanged, label_shift in zip(inputs_unchanged, labels_after_shift):
            assert input_unchanged == label_shift
        # Make sure that MASK index is correct in the inputs fed into the model.
        mask_inputs = inputs[indices_replaced]
        for input in mask_inputs:
            assert input == mask_idx

    def test_dynamic_masking(self):
        mask_idx = self.embedder._mask_id
        tokenizer_name = "roberta-large"
        # Generate 10 random input/label combinations and ensure that the indices make sense
        # in that the masked indices are approx 0.15 of the input, and indices_replace and indices_random
        # are subsets of the masked_indices and mutually exclusive.
        masked_ratio = []
        for i in range(10):
            length = random.randint(1, 510)
            orig_inputs = torch.LongTensor(
                [[4] + [random.randint(0, 10000) for x in range(length)] + [5]]
            )
            orig_labels = copy.deepcopy(orig_inputs)
            _, _, indices_replaced, indices_random, masked_indices, _ = self.MLMTask.mlm_dynamic_masking(
                orig_inputs, orig_labels, mask_idx, tokenizer_name, self.sent_encoder
            )
            indices_random = indices_random[0].nonzero().view(-1).numpy().tolist()
            indices_replaced = indices_replaced[0].nonzero().view(-1).numpy().tolist()
            indices_replaced_random = indices_random + indices_replaced
            # Make sure that indices_replaced & indices_random are mutually exclusive
            assert (
                len(indices_replaced_random) == (len(indices_random + indices_replaced))
                and len(set(indices_random).intersection(indices_replaced)) == 0
            )
            # Make sure that indices replaced and indices random are in the set of masked_indices
            masked_indices = masked_indices[0].nonzero().view(-1).numpy().tolist()
            assert set(indices_replaced_random).issubset(set(masked_indices))
            # Make sure that the masking approximately masks ~15% of the input.
            masked_ratio.append(float(len(masked_indices)) / float(len(orig_inputs[0])))
        assert np.mean(masked_ratio) >= 0.14 and np.mean(masked_ratio) <= 0.16
