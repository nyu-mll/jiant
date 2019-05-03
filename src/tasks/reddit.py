"""Task definitions for reddit."""
import codecs
import logging as log
import os
from typing import Any, Dict, Iterable, List, Sequence, Type

from allennlp.data import Instance, Token
from allennlp.data.fields import LabelField, MetadataField, TextField
from allennlp.training.metrics import Average

from ..utils.data_loaders import process_sentence
from ..utils.utils import truncate
from .registry import register_task
from .tasks import PairClassificationTask, RankingTask, sentence_to_text_field


@register_task("reddit", rel_path="Reddit_2008/")
@register_task("reddit_dummy", rel_path="Reddit_2008_TestSample/")
@register_task("reddit_3.4G", rel_path="Reddit_3.4G/")
@register_task("reddit_13G", rel_path="Reddit_13G/")
@register_task("reddit_softmax", rel_path="Reddit_2008/")
class RedditTask(RankingTask):
    """ Task class for Reddit data.  """

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super().__init__(name, **kw)
        self.scorer1 = Average()  # CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.files_by_split = {
            split: os.path.join(path, "%s.csv" % split) for split in ["train", "val", "test"]
        }
        self.max_seq_len = max_seq_len

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.

        Split should be one of 'train', 'val', or 'test'.
        """
        return self.load_data(self.files_by_split[split])

    def load_data(self, path):
        """ Load data """
        with open(path, "r") as txt_fh:
            for row in txt_fh:
                row = row.strip().split("\t")
                if len(row) < 4 or not row[2] or not row[3]:
                    continue
                sent1 = process_sentence(self._tokenizer_name, row[2], self.max_seq_len)
                sent2 = process_sentence(self._tokenizer_name, row[3], self.max_seq_len)
                targ = 1
                yield (sent1, sent2, targ)

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for sent1, sent2, _ in self.load_data(path):
                yield sent1
                yield sent2

    def count_examples(self):
        """ Compute here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for line in open(split_path))
        self.example_counts = example_counts

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(input1, input2, labels):
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            # d['sent1_str'] = MetadataField(" ".join(input1[1:-1]))
            d["input2"] = sentence_to_text_field(input2, indexers)
            # d['sent2_str'] = MetadataField(" ".join(input2[1:-1]))
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            return Instance(d)

        for sent1, sent2, trg in split:
            yield _make_instance(sent1, sent2, trg)

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}


@register_task("reddit_pair_classif", rel_path="Reddit/")
@register_task("reddit_pair_classif_3.4G", rel_path="Reddit_3.4G/")
class RedditPairClassificationTask(PairClassificationTask):
    """ Task class for Reddit data.  """

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super().__init__(name, n_classes=2, **kw)
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.files_by_split = {
            split: os.path.join(path, "%s.csv" % split) for split in ["train", "val", "test"]
        }
        self.max_seq_len = max_seq_len

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.

        Split should be one of 'train', 'val', or 'test'.
        """
        return self.load_data(self.files_by_split[split])

    def load_data(self, path):
        """ Load data """
        with open(path, "r") as txt_fh:
            for row in txt_fh:
                row = row.strip().split("\t")
                if len(row) < 4 or not row[2] or not row[3]:
                    continue
                sent1 = process_sentence(self._tokenizer_name, row[2], self.max_seq_len)
                sent2 = process_sentence(self._tokenizer_name, row[3], self.max_seq_len)
                targ = 1
                yield (sent1, sent2, targ)

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for sent1, sent2, _ in self.load_data(path):
                yield sent1
                yield sent2

    def count_examples(self):
        """ Compute here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for line in open(split_path))
        self.example_counts = example_counts

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(input1, input2, labels):
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            # d['sent1_str'] = MetadataField(" ".join(input1[1:-1]))
            d["input2"] = sentence_to_text_field(input2, indexers)
            # d['sent2_str'] = MetadataField(" ".join(input2[1:-1]))
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            return Instance(d)

        for sent1, sent2, trg in split:
            yield _make_instance(sent1, sent2, trg)

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}


@register_task("mt_pair_classif", rel_path="wmt14_en_de_local/")
@register_task("mt_pair_classif_dummy", rel_path="wmt14_en_de_mini/")
class MTDataPairClassificationTask(RedditPairClassificationTask):
    """ Task class for MT data pair classification using standard setup.
        RedditPairClassificationTask and MTDataPairClassificationTask are same tasks with different data
    """

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super().__init__(path, max_seq_len, name, **kw)
        self.files_by_split = {
            split: os.path.join(path, "%s.txt" % split) for split in ["train", "val", "test"]
        }

    def load_data(self, path):
        """ Load data """
        with codecs.open(path, "r", "utf-8", errors="ignore") as txt_fh:
            for row in txt_fh:
                row = row.strip().split("\t")
                if len(row) < 2 or not row[0] or not row[1]:
                    continue
                sent1 = process_sentence(self._tokenizer_name, row[0], self.max_seq_len)
                sent2 = process_sentence(self._tokenizer_name, row[1], self.max_seq_len)
                targ = 1
                yield (sent1, sent2, targ)

    def count_examples(self):
        """ Compute here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(
                1 for line in codecs.open(split_path, "r", "utf-8", errors="ignore")
            )
        self.example_counts = example_counts
