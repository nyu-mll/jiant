import csv
import os
import os.path
from pkg_resources import resource_filename
import shutil
import tempfile
import unittest
from unittest import mock

import jiant.tasks.tasks as tasks
from jiant.utils.config import params_from_file
from jiant.preprocess import get_task_without_loading_data, build_indexers, get_vocab


class TestProprocess(unittest.TestCase):
    def setUp(self):
        self.HOCON1 = """
            pretrain_tasks = mnli
            target_tasks = qqp
            tokenizer = bert-large-cased
            input_module = bert-large-cased
        """
        self.HOCON2 = """
            pretrain_task s= mnli
            target_tasks = qqp
            input_module = glove
            tokenizer = MosesTokenizer
        """
        self.HOCON3 = """
            pretrain_task s= mnli
            target_tasks = qqp
            input_module = openai-gpt
            tokenizer = openai-gpt
        """
        self.HOCON4 = """
            pretrain_tasks = mnli
            target_tasks = qqp
            tokenizer = bert-large-cased
            input_module = bert-base-cased
        """
        self.DEFAULTS_PATH = resource_filename(
            "jiant", "config/defaults.conf"
        )  # To get other required values.
        self.params1 = params_from_file(self.DEFAULTS_PATH, self.HOCON1)

    def test_get_task_without_loading_data(self):
        # Test task laoding.
        tasks = get_task_without_loading_data("mnli", self.params1)
        assert tasks.name == "mnli"

    def test_build_indexers(self):
        self.params2 = params_from_file(self.DEFAULTS_PATH, self.HOCON2)
        self.params3 = params_from_file(self.DEFAULTS_PATH, self.HOCON3)
        self.params4 = params_from_file(self.DEFAULTS_PATH, self.HOCON4)
        indexer = build_indexers(self.params1)
        len(indexer) == 1 and list(indexer.keys())[0] == "bert_cased"
        indexer = build_indexers(self.params2)
        len(indexer) == 1 and list(indexer.keys())[0] == "words"
        indexer = build_indexers(self.params3)
        len(indexer) == 1 and list(indexer.keys())[0] == "openai_gpt"
        with self.assertRaises(AssertionError) as error:
            # BERT model and tokenizer must be equal, so this should throw an error.
            indexer = build_indexers(self.params4)

    def test_build_vocab(self):
        word2freq = {"first": 100, "second": 5, "third": 10}
        char2freq = {"a": 10, "b": 100, "c": 30}
        max_vocab_size = {"word": 2, "char": 3}
        vocab = get_vocab(word2freq, char2freq, max_vocab_size)
        assert len(vocab.get_index_to_token_vocabulary("tokens")) == 6
        assert set(vocab.get_index_to_token_vocabulary("tokens").values()) == set(
            ["@@PADDING@@", "@@UNKNOWN@@", "<SOS>", "<EOS>", "first", "third"]
        )
        assert len(vocab.get_index_to_token_vocabulary("chars")) == 5
        assert set(vocab.get_index_to_token_vocabulary("chars").values()) == set(
            ["@@PADDING@@", "@@UNKNOWN@@", "a", "b", "c"]
        )
