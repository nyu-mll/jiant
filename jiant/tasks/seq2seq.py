"""Task definitions for sequence-to-sequence tasks."""
import codecs
import collections
import math
import os
from typing import Iterable, List, Sequence, Type

from allennlp.data import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.training.metrics import Average

from ..utils.data_loaders import process_sentence
from .registry import register_task
from .tasks import (
    UNK_TOK_ALLENNLP,
    UNK_TOK_ATOMIC,
    SequenceGenerationTask,
    atomic_tokenize,
    sentence_to_text_field,
)


@register_task("seg_wix", rel_path="seg/wix/", max_targ_v_size=200)
class CharSeq2SeqTask(SequenceGenerationTask):
    """Character-based Sequence-to-sequence Task"""

    def __init__(self, path, max_seq_len, max_targ_v_size, name, **kw):
        """ """
        super().__init__(name, **kw)
        self.scorer1 = Average()
        self.scorer2 = Average()
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_perplexity" % self.name
        self.val_metric_decreases = True
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tokens"
        self.max_targ_v_size = max_targ_v_size
        self.target_indexer = {"words": SingleIdTokenIndexer(namespace=self._label_namespace)}
        self.files_by_split = {
            split: os.path.join(path, "%s.tsv" % split) for split in ["train", "val", "test"]
        }

    def load_data(self):
        # Data is exposed as iterable: no preloading
        pass

    def get_split_text(self, split: str):
        """
        Get split text as iterable of records.
        Split should be one of 'train', 'val', or 'test'.
        """
        return self.get_data_iter(self.files_by_split[split])

    def get_all_labels(self) -> List[str]:
        """ Build character vocabulary and return it as a list """
        word2freq = collections.Counter()
        for split in ["train", "val"]:
            for _, sent in self.get_data_iter(self.files_by_split[split]):
                for word in sent:
                    word2freq[word] += 1
        return [w for w, _ in word2freq.most_common(self.max_targ_v_size)]

    def get_data_iter(self, path):
        """ Load data """
        with codecs.open(path, "r", "utf-8", errors="ignore") as txt_fh:
            for row in txt_fh:
                row = row.strip().split("\t")
                if len(row) < 2 or not row[0] or not row[1]:
                    continue
                # "SplitChars" is the tokenizer here.
                src_sent = process_sentence("SplitChars", row[0], self.max_seq_len)
                tgt_sent = process_sentence("SplitChars", row[2], self.max_seq_len)
                yield (src_sent, tgt_sent)

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            yield from self.get_data_iter(path)

    def count_examples(self):
        """ Compute here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(
                1 for _ in codecs.open(split_path, "r", "utf-8", errors="ignore")
            )
        self.example_counts = example_counts

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(input_, target):
            d = {
                "inputs": sentence_to_text_field(input_, indexers),
                "targs": sentence_to_text_field(target, self.target_indexer),
            }
            return Instance(d)

        for sent1, sent2 in split:
            yield _make_instance(sent1, sent2)

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        avg_nll = self.scorer1.get_metric(reset)
        avg_acc = self.scorer2.get_metric(reset)
        return {"perplexity": math.exp(avg_nll), "accuracy": avg_acc}

    def update_metrics(self, logits, labels, tagmask=None):
        # TODO(Katharina): NLL for scorer1 and acc for scorer2
        self.scorer1(mean_squared_error(logits, labels))  # update average MSE
        self.scorer2(logits, labels)
        return
