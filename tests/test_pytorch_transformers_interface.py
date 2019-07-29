import unittest
from unittest import mock
import torch
from jiant.pytorch_transformers_interface.modules import BertEmbedderModule, XLNetEmbedderModule


class TestPytorchTransformersInterface(unittest.TestCase):
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
        inp = torch.Tensor([[5, 8, 3, 9, 10, 11, 3], [5, 8, 9, 3, 10, 3, 7]])
        output = bert_model.get_seg_ids(bert_model, inp)
        assert torch.all(
            torch.eq(output, torch.Tensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 0]]))
        )

        # [CLS] 8 9 [SEP]
        # [CLS] 8 [SEP] [PAD]
        inp = torch.Tensor([[5, 8, 9, 3], [5, 9, 3, 7]])
        output = bert_model.get_seg_ids(bert_model, inp)
        assert torch.all(torch.eq(output, torch.Tensor([[0, 0, 0, 0], [0, 0, 0, 0]])))

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
        inp = torch.Tensor([[8, 3, 9, 10, 11, 3, 5], [8, 3, 9, 10, 3, 5, 7]])
        output = xlnet_model.get_seg_ids(xlnet_model, inp)
        assert torch.all(
            torch.eq(output, torch.Tensor([[0, 3, 1, 1, 1, 3, 2], [0, 3, 1, 1, 3, 2, 0]]))
        )

        # 8 9 10 [SEP] [CLS]
        # 8 9 [SEP] [CLS] [PAD]
        inp = torch.Tensor([[8, 9, 10, 3, 5], [8, 9, 3, 5, 7]])
        output = xlnet_model.get_seg_ids(xlnet_model, inp)
        assert torch.all(torch.eq(output, torch.Tensor([[0, 0, 0, 3, 2], [0, 0, 3, 2, 0]])))
