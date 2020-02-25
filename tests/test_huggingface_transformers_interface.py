import unittest
from unittest import mock
import torch
import copy
from jiant.huggingface_transformers_interface.modules import (
    HuggingfaceTransformersEmbedderModule,
    BertEmbedderModule,
    RobertaEmbedderModule,
    AlbertEmbedderModule,
    XLNetEmbedderModule,
    OpenAIGPTEmbedderModule,
    GPT2EmbedderModule,
    TransfoXLEmbedderModule,
    XLMEmbedderModule,
)


class TestHuggingfaceTransformersInterface(unittest.TestCase):
    def test_bert_apply_boundary_tokens(self):
        s1 = ["A", "B", "C"]
        s2 = ["D", "E"]
        self.assertListEqual(
            BertEmbedderModule.apply_boundary_tokens(s1), ["[CLS]", "A", "B", "C", "[SEP]"]
        )
        self.assertListEqual(
            BertEmbedderModule.apply_boundary_tokens(s1, s2),
            ["[CLS]", "A", "B", "C", "[SEP]", "D", "E", "[SEP]"],
        )

    def test_roberta_apply_boundary_tokens(self):
        s1 = ["A", "B", "C"]
        s2 = ["D", "E"]
        self.assertListEqual(
            RobertaEmbedderModule.apply_boundary_tokens(s1), ["<s>", "A", "B", "C", "</s>"]
        )
        self.assertListEqual(
            RobertaEmbedderModule.apply_boundary_tokens(s1, s2),
            ["<s>", "A", "B", "C", "</s>", "</s>", "D", "E", "</s>"],
        )

    def test_albert_apply_boundary_tokens(self):
        s1 = ["A", "B", "C"]
        s2 = ["D", "E"]
        self.assertListEqual(
            AlbertEmbedderModule.apply_boundary_tokens(s1), ["[CLS]", "A", "B", "C", "[SEP]"]
        )
        self.assertListEqual(
            AlbertEmbedderModule.apply_boundary_tokens(s1, s2),
            ["[CLS]", "A", "B", "C", "[SEP]", "D", "E", "[SEP]"],
        )

    def test_xlnet_apply_boundary_tokens(self):
        s1 = ["A", "B", "C"]
        s2 = ["D", "E"]
        self.assertListEqual(
            XLNetEmbedderModule.apply_boundary_tokens(s1), ["A", "B", "C", "<sep>", "<cls>"]
        )
        self.assertListEqual(
            XLNetEmbedderModule.apply_boundary_tokens(s1, s2),
            ["A", "B", "C", "<sep>", "D", "E", "<sep>", "<cls>"],
        )

    def test_gpt_apply_boundary_tokens(self):
        s1 = ["A", "B", "C"]
        s2 = ["D", "E"]
        self.assertListEqual(
            OpenAIGPTEmbedderModule.apply_boundary_tokens(s1),
            ["<start>", "A", "B", "C", "<extract>"],
        )
        self.assertListEqual(
            OpenAIGPTEmbedderModule.apply_boundary_tokens(s1, s2),
            ["<start>", "A", "B", "C", "<delim>", "D", "E", "<extract>"],
        )

    def test_xlm_apply_boundary_tokens(self):
        s1 = ["A", "B", "C"]
        s2 = ["D", "E"]
        self.assertListEqual(
            XLMEmbedderModule.apply_boundary_tokens(s1), ["</s>", "A", "B", "C", "</s>"]
        )
        self.assertListEqual(
            XLMEmbedderModule.apply_boundary_tokens(s1, s2),
            ["</s>", "A", "B", "C", "</s>", "D", "E", "</s>"],
        )

    def test_correct_sent_indexing(self):
        model = mock.Mock()
        model._pad_id = 7
        model._unk_id = 10
        model.max_pos = None
        model.tokenizer_required = "correct_tokenizer"
        model.correct_sent_indexing = HuggingfaceTransformersEmbedderModule.correct_sent_indexing

        allenNLP_indexed = torch.LongTensor([[7, 10, 5, 11, 1, 13, 5], [7, 10, 11, 5, 1, 5, 0]])

        expected_ids = torch.LongTensor([[5, 8, 3, 9, 10, 11, 3], [5, 8, 9, 3, 10, 3, 7]])
        expected_mask = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]])

        # test the required_tokenizer assertion(bad case)
        assertionerror_found = False
        sent = {"wrong_tokenizer": copy.deepcopy(allenNLP_indexed)}
        try:
            model.correct_sent_indexing(model, sent)
        except AssertionError:
            assertionerror_found = True
        assert assertionerror_found

        # test the required_tokenizer, unk_id assertion(good case)
        # test the result correctness
        sent = {"correct_tokenizer": copy.deepcopy(allenNLP_indexed)}
        ids, input_mask = model.correct_sent_indexing(model, sent)
        assert torch.all(torch.eq(ids, expected_ids))
        assert torch.all(torch.eq(input_mask, expected_mask))

        # test the unk_id assertion(bad case)
        model._unk_id = None
        assertionerror_found = False
        sent = {"correct_tokenizer": copy.deepcopy(allenNLP_indexed)}
        try:
            model.correct_sent_indexing(model, sent)
        except AssertionError:
            assertionerror_found = True
        assert assertionerror_found
        model._unk_id = 10

        # test the max input length assertion(bad case)
        model.max_pos = 6
        assertionerror_found = False
        sent = {"correct_tokenizer": copy.deepcopy(allenNLP_indexed)}
        try:
            model.correct_sent_indexing(model, sent)
        except AssertionError:
            assertionerror_found = True
        assert assertionerror_found

        # test the max input length assertion(good case)
        model.max_pos = 7
        sent = {"correct_tokenizer": copy.deepcopy(allenNLP_indexed)}
        ids, input_mask = model.correct_sent_indexing(model, sent)
        assert torch.all(torch.eq(ids, expected_ids))
        assert torch.all(torch.eq(input_mask, expected_mask))

    def test_bert_seg_ids(self):
        bert_model = mock.Mock()
        bert_model._sep_id = 3
        bert_model._cls_id = 5
        bert_model._pad_id = 7
        bert_model._SEG_ID_CLS = None
        bert_model._SEG_ID_SEP = None
        bert_model.get_seg_ids = BertEmbedderModule.get_seg_ids

        # [CLS] 8 [SEP] 9 10 11 [SEP]
        # [CLS] 8 9 [SEP] 10 [SEP] [PAD]
        inp = torch.LongTensor([[5, 8, 3, 9, 10, 11, 3], [5, 8, 9, 3, 10, 3, 7]])
        mask = inp != bert_model._pad_id
        output = bert_model.get_seg_ids(bert_model, inp, mask.long())
        assert torch.all(
            torch.eq(output, torch.LongTensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0]]))
        )

        # [CLS] 8 9 [SEP]
        # [CLS] 8 [SEP] [PAD]
        inp = torch.LongTensor([[5, 8, 9, 3], [5, 9, 3, 7]])
        mask = inp != bert_model._pad_id
        output = bert_model.get_seg_ids(bert_model, inp, mask.long())
        assert torch.all(torch.eq(output, torch.LongTensor([[0, 0, 0, 0], [0, 0, 0, 0]])))

    def test_xlnet_seg_ids(self):
        xlnet_model = mock.Mock()
        xlnet_model._sep_id = 3
        xlnet_model._cls_id = 5
        xlnet_model._pad_id = 7
        xlnet_model._SEG_ID_CLS = 2
        xlnet_model._SEG_ID_SEP = 3
        xlnet_model.get_seg_ids = XLNetEmbedderModule.get_seg_ids

        # 8 [SEP] 9 10 11 [SEP] [CLS]
        # 8 [SEP] 9 10 [SEP] [CLS] [PAD]
        inp = torch.LongTensor([[8, 3, 9, 10, 11, 3, 5], [8, 3, 9, 10, 3, 5, 7]])
        mask = inp != xlnet_model._pad_id
        output = xlnet_model.get_seg_ids(xlnet_model, inp, mask.long())
        assert torch.all(
            torch.eq(output, torch.LongTensor([[0, 3, 1, 1, 1, 3, 2], [0, 3, 1, 1, 3, 2, 0]]))
        )

        # 8 9 10 [SEP] [CLS]
        # 8 9 [SEP] [CLS] [PAD]
        inp = torch.LongTensor([[8, 9, 10, 3, 5], [8, 9, 3, 5, 7]])
        mask = inp != xlnet_model._pad_id
        output = xlnet_model.get_seg_ids(xlnet_model, inp, mask.long())
        assert torch.all(torch.eq(output, torch.LongTensor([[0, 0, 0, 3, 2], [0, 0, 3, 2, 0]])))
