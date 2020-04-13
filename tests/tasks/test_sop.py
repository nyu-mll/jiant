import unittest
import tempfile
import os
import random
import copy
import numpy as np
from unittest import mock
from jiant.tasks.registry import REGISTRY
from jiant.modules.sentence_encoder import SentenceEncoder
from jiant.huggingface_transformers_interface.modules import HuggingfaceTransformersEmbedderModule
from jiant.modules.simple_modules import NullPhraseLayer


class TestSOP(unittest.TestCase):
    def setUp(self):
        cls, _, kw = REGISTRY["wikipedia_corpus_sop"]
        self.temp_dir = tempfile.mkdtemp()
        self.max_seq_len = 24
        self.SOPTask = cls(
            os.path.join("wikipedia_corpus_sop"),
            max_seq_len=self.max_seq_len,
            name="wikipedia_corpus_sop",
            tokenizer_name="roberta-large",
            **kw,
        )
        os.mkdir(os.path.join(self.temp_dir, "wikipedia_corpus_sop"))
        self.train_path = os.path.join(self.temp_dir, "wikipedia_corpus_sop", "train.txt")
        with open(self.train_path, "w") as write_fn:
            write_fn.write("1Let's see if SOP works. \n")
            write_fn.write("1SOP is one of two pretraining objectives. \n")
            write_fn.write("1The other one is MLM.")
            write_fn.write("=========END OF ARTICLE======== \n")
            write_fn.write("2NLP is pretty cool.\n")
            write_fn.write("2An area of focus in the NYU lab is transfer learning.\n")
            write_fn.write("2There's some pretty cool stuff.")
            write_fn.write("=========END OF ARTICLE======== \n")
            write_fn.close()

    def test_sop_preprocessing(self):
        train_examples = list(self.SOPTask.get_data_iter(self.train_path))
        for example in train_examples:
            assert example[0][0] == example[1][0]  # should be same number.
            assert "=" not in "".join(
                example[0] + example[1]
            )  # Make sure END OF ARTICLE is not included as an example.
            assert len(example[0]) + len(example[1]) <= self.max_seq_len - 3
