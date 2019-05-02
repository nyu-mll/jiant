"""Task definitions for machine translation tasks."""
import codecs
import collections
import logging as log
import math
import os

import allennlp.common.util as allennlp_util
from allennlp.training.metrics import Average
from allennlp.data.token_indexers import SingleIdTokenIndexer

from allennlp.data import Instance, Token

from ..utils.data_loaders import process_sentence
from ..utils.utils import truncate

from typing import Iterable, Sequence, List, Dict, Any, Type

from .tasks import SequenceGenerationTask
from .tasks import sentence_to_text_field, atomic_tokenize
from .tasks import UNK_TOK_ALLENNLP, UNK_TOK_ATOMIC
from .registry import register_task

# TODO: remove dummy / debug tasks


@register_task('wmt_debug', rel_path='wmt_debug/', max_targ_v_size=5000)
@register_task('wmt17_en_ru', rel_path='wmt17_en_ru/', max_targ_v_size=20000)
@register_task('wmt14_en_de', rel_path='wmt14_en_de/', max_targ_v_size=20000)
class MTTask(SequenceGenerationTask):
    '''Machine Translation Task'''

    def __init__(self, path, max_seq_len, max_targ_v_size, name, **kw):
        ''' '''
        super().__init__(name, **kw)
        self.scorer1 = Average()
        self.scorer2 = Average()
        self.scorer3 = Average()
        self.scorers = [self.scorer1, self.scorer2, self.scorer3]
        self.val_metric = "%s_perplexity" % self.name
        self.val_metric_decreases = True
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tokens"
        self.max_targ_v_size = max_targ_v_size
        self.target_indexer = {
            "words": SingleIdTokenIndexer(
                namespace=self._label_namespace)}
        self.files_by_split = {split: os.path.join(path, "%s.txt" % split) for
                               split in ["train", "val", "test"]}

    def get_split_text(self, split: str):
        ''' Get split text as iterable of records.

        Split should be one of 'train', 'val', or 'test'.
        '''
        return self.load_data(self.files_by_split[split])

    def get_all_labels(self) -> List[str]:
        ''' Build vocabulary and return it as a list '''
        word2freq = collections.Counter()
        for split in ["train", "val"]:
            for _, sent in self.load_data(self.files_by_split[split]):
                for word in sent:
                    word2freq[word] += 1
        return [w for w, _ in word2freq.most_common(self.max_targ_v_size)]

    def load_data(self, path):
        ''' Load data '''
        with codecs.open(path, 'r', 'utf-8', errors='ignore') as txt_fh:
            for row in txt_fh:
                row = row.strip().split('\t')
                if len(row) < 2 or not row[0] or not row[1]:
                    continue
                src_sent = process_sentence(
                    self._tokenizer_name, row[0], self.max_seq_len)
                # Currently: force Moses tokenization on targets
                tgt_sent = process_sentence(
                    "MosesTokenizer", row[1], self.max_seq_len)
                yield (src_sent, tgt_sent)

    def get_sentences(self) -> Iterable[Sequence[str]]:
        ''' Yield sentences, used to compute vocabulary. '''
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            yield from self.load_data(path)

    def count_examples(self):
        ''' Compute here b/c we're streaming the sentences. '''
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(
                1 for line in codecs.open(
                    split_path, 'r', 'utf-8', errors='ignore'))
        self.example_counts = example_counts

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        def _make_instance(input, target):
            d = {}
            d["inputs"] = sentence_to_text_field(input, indexers)
            d["targs"] = sentence_to_text_field(target, self.target_indexer)
            return Instance(d)

        for sent1, sent2 in split:
            yield _make_instance(sent1, sent2)

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        avg_nll = self.scorer1.get_metric(reset)
        unk_ratio_macroavg = self.scorer3.get_metric(reset)
        return {
            'perplexity': math.exp(avg_nll),
            'bleu_score': 0,
            'unk_ratio_macroavg': unk_ratio_macroavg}


@register_task('reddit_s2s', rel_path='Reddit/', max_targ_v_size=0)
@register_task('reddit_s2s_3.4G', rel_path='Reddit_3.4G/', max_targ_v_size=0)
class RedditSeq2SeqTask(MTTask):
    ''' Task for seq2seq using reddit data

    Note: max_targ_v_size doesn't do anything here b/c the
    target is in English'''

    def __init__(self, path, max_seq_len, max_targ_v_size, name, **kw):
        super().__init__(path=path, max_seq_len=max_seq_len,
                         max_targ_v_size=max_targ_v_size, name=name,
                         **kw)
        self._label_namespace = None
        self.target_indexer = {"words": SingleIdTokenIndexer("tokens")}
        self.files_by_split = {"train": os.path.join(path, "train.csv"),
                               "val": os.path.join(path, "val.csv"),
                               "test": os.path.join(path, "test.csv")}

    def load_data(self, path):
        ''' Load data '''
        with codecs.open(path, 'r', 'utf-8', errors='ignore') as txt_fh:
            for row in txt_fh:
                row = row.strip().split('\t')
                if len(row) < 4 or not row[2] or not row[3]:
                    continue
                src_sent = process_sentence(
                    self._tokenizer_name, row[2], self.max_seq_len)
                tgt_sent = process_sentence(
                    self._tokenizer_name, row[3], self.max_seq_len)
                yield (src_sent, tgt_sent)

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        def _make_instance(input, target):
            d = {}
            d["inputs"] = sentence_to_text_field(input, indexers)
            d["targs"] = sentence_to_text_field(target, self.target_indexer)
            return Instance(d)

        for sent1, sent2 in split:
            yield _make_instance(sent1, sent2)


@register_task('wiki2_s2s', rel_path='WikiText2/', max_targ_v_size=0)
@register_task('wiki103_s2s', rel_path='WikiText103/', max_targ_v_size=0)
class Wiki103Seq2SeqTask(MTTask):
    ''' Skipthought objective on Wiki103 '''

    def __init__(self, path, max_seq_len, max_targ_v_size, name, **kw):
        ''' Note: max_targ_v_size does nothing here '''
        super().__init__(path, max_seq_len, max_targ_v_size, name, **kw)
        # for skip-thoughts setting, all source sentences are sentences that
        # followed by another sentence (which are all but the last one).
        # Similar for self.target_sentences
        self._nonatomic_toks = [UNK_TOK_ALLENNLP, '<unk>']
        self._label_namespace = None
        self.target_indexer = {"words": SingleIdTokenIndexer("tokens")}
        self.files_by_split = {"train": os.path.join(path, "train.sentences.txt"),
                               "val": os.path.join(path, "valid.sentences.txt"),
                               "test": os.path.join(path, "test.sentences.txt")}

    def load_data(self, path):
        ''' Load data '''
        nonatomic_toks = self._nonatomic_toks
        with codecs.open(path, 'r', 'utf-8', errors='ignore') as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                sent = atomic_tokenize(toks, UNK_TOK_ATOMIC, nonatomic_toks, self.max_seq_len,
                                       tokenizer_name=self._tokenizer_name)
                yield sent, []

    def get_num_examples(self, split_text):
        ''' Return number of examples in the result of get_split_text.

        Subclass can override this if data is not stored in column format.
        '''
        # pair setences# = sent# - 1
        return len(split_text) - 1

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process a language modeling split.

        Split is a single list of sentences here.
        '''
        def _make_instance(prev_sent, sent):
            d = {}
            d["inputs"] = sentence_to_text_field(prev_sent, indexers)
            d["targs"] = sentence_to_text_field(sent, self.target_indexer)
            return Instance(d)

        prev_sent = None
        for sent, _ in split:
            if prev_sent is None:
                prev_sent = sent
                continue
            yield _make_instance(prev_sent, sent)
            prev_sent = sent
