'''Define the tasks and code for loading their data.

- As much as possible, following the existing task hierarchy structure.
- When inheriting, be sure to write and call load_data.
- Set all text data as an attribute, task.sentences (List[List[str]])
- Each task's val_metric should be name_metric, where metric is returned by
get_metrics(): e.g. if task.val_metric = task_name + "_accuracy", then
task.get_metrics() should return {"accuracy": accuracy_val, ... }
'''
import codecs
import collections
import copy
import itertools
import json
import logging as log
import os

import numpy as np
import torch

from allennlp.training.metrics import CategoricalAccuracy, \
    BooleanAccuracy, F1Measure, Average
from ..allennlp_mods.correlation import Correlation, FastMatthews
from allennlp.data.token_indexers import SingleIdTokenIndexer

# Fields for instance processing
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, LabelField, \
    SpanField, ListField, MetadataField
from ..allennlp_mods.numeric_field import NumericField

from ..utils import utils
from ..utils.utils import truncate
from ..utils.data_loaders import load_tsv, process_sentence, load_diagnostic_tsv
from ..utils.tokenizers import get_tokenizer

from typing import Iterable, Sequence, List, Dict, Any, Type

UNK_TOK_ALLENNLP = "@@UNKNOWN@@"
UNK_TOK_ATOMIC = "UNKNOWN"  # an unk token that won't get split by tokenizers

from .registry import register_task, REGISTRY  # global task registry

def sentence_to_text_field(sent: Sequence[str], indexers: Any):
    ''' Helper function to map a sequence of tokens into a sequence of
    AllenNLP Tokens, then wrap in a TextField with the given indexers '''
    return TextField(list(map(Token, sent)), token_indexers=indexers)


def atomic_tokenize(sent: str, atomic_tok: str, nonatomic_toks: List[str], max_seq_len: int):
    ''' Replace tokens that will be split by tokenizer with a
    placeholder token. Tokenize, and then substitute the placeholder
    with the *first* nonatomic token in the list. '''
    for nonatomic_tok in nonatomic_toks:
        sent = sent.replace(nonatomic_tok, atomic_tok)
    sent = process_sentence(sent, max_seq_len)
    sent = [nonatomic_toks[0] if t == atomic_tok else t for t in sent]
    return sent


def process_single_pair_task_split(split, indexers, is_pair=True, classification=True):
    '''
    Convert a dataset of sentences into padded sequences of indices. Shared
    across several classes.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)

    Returns:
        - instances (list[Instance]): a list of AllenNLP Instances with fields
    '''
    def _make_instance(input1, input2, labels, idx):
        d = {}
        d["input1"] = sentence_to_text_field(input1, indexers)
        d['sent1_str'] = MetadataField(" ".join(input1[1:-1]))
        if input2:
            d["input2"] = sentence_to_text_field(input2, indexers)
            d['sent2_str'] = MetadataField(" ".join(input2[1:-1]))
        if classification:
            d["labels"] = LabelField(labels, label_namespace="labels",
                                     skip_indexing=True)
        else:
            d["labels"] = NumericField(labels)

        d["idx"] = LabelField(idx, label_namespace="idxs",
                              skip_indexing=True)

        return Instance(d)

    split = list(split)
    if not is_pair:  # dummy iterator for input2
        split[1] = itertools.repeat(None)
    if len(split) < 4:  # counting iterator for idx
        assert len(split) == 3
        split.append(itertools.count())

    # Map over columns: input2, (input2), labels, idx
    instances = map(_make_instance, *split)
    #  return list(instances)
    return instances  # lazy iterator


class Task(object):
    '''Generic class for a task

    Methods and attributes:
        - load_data: load dataset from a path and create splits
        - truncate: truncate data to be at most some length
        - get_metrics:

    Outside the task:
        - process: pad and indexify data given a mapping
        - optimizer
    '''
    def __init__(self, name, tokenizer_name):
        self.name = name
        assert self.tokenizer_is_supported(tokenizer_name)
        self._tokenizer_name = tokenizer_name

    def load_data(self, path, max_seq_len):
        ''' Load data from path and create splits. '''
        raise NotImplementedError

    def truncate(self, max_seq_len, sos_tok, eos_tok):
        ''' Shorten sentences to max_seq_len and add sos and eos tokens. '''
        raise NotImplementedError

    def get_sentences(self) -> Iterable[Sequence[str]]:
        ''' Yield sentences, used to compute vocabulary. '''
        yield from self.sentences

    def count_examples(self, splits=['train', 'val', 'test']):
        ''' Count examples in the dataset. '''
        self.example_counts = {}
        for split in splits:
            st = self.get_split_text(split)
            count = self.get_num_examples(st)
            self.example_counts[split] = count

    def tokenizer_is_supported(self, tokenizer_name):
        ''' Check if the tokenizer is supported for this task. '''
        return get_tokenizer(tokenizer_name) is not None

    @property
    def tokenizer_name(self):
        return self._tokenizer_name

    @property
    def n_train_examples(self):
        return self.example_counts['train']

    @property
    def n_val_examples(self):
        return self.example_counts['val']

    def get_split_text(self, split: str):
        ''' Get split text, typically as list of columns.

        Split should be one of 'train', 'val', or 'test'.
        '''
        return getattr(self, '%s_data_text' % split)

    def get_num_examples(self, split_text):
        ''' Return number of examples in the result of get_split_text.

        Subclass can override this if data is not stored in column format.
        '''
        return len(split_text[0])

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict:
        ''' Get metrics specific to the task. '''
        raise NotImplementedError


class ClassificationTask(Task):
    ''' General classification task '''
    pass


class RegressionTask(Task):
    ''' General regression task '''
    pass


class SingleClassificationTask(ClassificationTask):
    ''' Generic sentence pair classification '''

    def __init__(self, name, n_classes, **kw):
        super().__init__(name, **kw)
        self.n_classes = n_classes
        self.scorer1 = CategoricalAccuracy()
        self.scorer2 = None
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False

    def truncate(self, max_seq_len, sos_tok="<SOS>", eos_tok="<EOS>"):
        self.train_data_text = [truncate(self.train_data_text[0], max_seq_len,
                                         sos_tok, eos_tok), self.train_data_text[1]]
        self.val_data_text = [truncate(self.val_data_text[0], max_seq_len,
                                       sos_tok, eos_tok), self.val_data_text[1]]
        self.test_data_text = [truncate(self.test_data_text[0], max_seq_len,
                                        sos_tok, eos_tok), self.test_data_text[1]]

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        return {'accuracy': acc}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        return process_single_pair_task_split(split, indexers, is_pair=False)


class PairClassificationTask(ClassificationTask):
    ''' Generic sentence pair classification '''

    def __init__(self, name, n_classes, **kw):
        super().__init__(name, **kw)
        assert n_classes > 0
        self.n_classes = n_classes
        self.scorer1 = CategoricalAccuracy()
        self.scorer2 = None
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        return {'accuracy': acc}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        return process_single_pair_task_split(split, indexers, is_pair=True)


class PairRegressionTask(RegressionTask):
    ''' Generic sentence pair classification '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.n_classes = 1
        self.scorer1 = Average()  # for average MSE
        self.scorer2 = None
        self.val_metric = "%s_mse" % self.name
        self.val_metric_decreases = True

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        mse = self.scorer1.get_metric(reset)
        return {'mse': mse}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        return process_single_pair_task_split(split, indexers, is_pair=True,
                                              classification=False)


class PairOrdinalRegressionTask(RegressionTask):
    ''' Generic sentence pair ordinal regression.
        Currently just doing regression but added new class
        in case we find a good way to implement ordinal regression with NN'''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.n_classes = 1
        self.scorer1 = Average()  # for average MSE
        self.scorer2 = Correlation('spearman')
        self.val_metric = "%s_1-mse" % self.name
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        mse = self.scorer1.get_metric(reset)
        spearmanr = self.scorer2.get_metric(reset)
        return {'1-mse': 1 - mse,
                'mse': mse,
                'spearmanr': spearmanr}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        return process_single_pair_task_split(split, indexers, is_pair=True,
                                              classification=False)


class SequenceGenerationTask(Task):
    ''' Generic sentence generation task '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.scorer1 = Average()  # for average BLEU or something
        self.scorer2 = None
        self.val_metric = "%s_bleu" % self.name
        self.val_metric_decreases = False
        log.warning("BLEU scoring is turned off (current code in progress)."
                    "Please use outputed prediction files to score offline")

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        bleu = self.scorer1.get_metric(reset)
        return {'bleu': bleu}


class RankingTask(Task):
    ''' Generic sentence ranking task, given some input '''
    pass


@register_task('sst', rel_path='SST-2/')
class SSTTask(SingleClassificationTask):
    ''' Task class for Stanford Sentiment Treebank.  '''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' '''
        super(SSTTask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]

    def load_data(self, path, max_seq_len):
        ''' Load data '''
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=0, s2_idx=None, label_idx=1, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, 'dev.tsv'), max_seq_len,
                            s1_idx=0, s2_idx=None, label_idx=1, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=None, has_labels=False, return_indices=True, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SST data.")


@register_task('cola', rel_path='CoLA/')
class CoLATask(SingleClassificationTask):
    '''Class for Warstdadt acceptability task'''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' '''
        super(CoLATask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        #self.scorer1 = Average()
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=None, label_idx=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=None, label_idx=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=None, has_labels=False, return_indices=True, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading CoLA.")

    def get_metrics(self, reset=False):
        return {'mcc': self.scorer1.get_metric(reset),
                'accuracy': self.scorer2.get_metric(reset)}


@register_task('qqp', rel_path='QQP/')
@register_task('qqp-alt', rel_path='QQP/')  # second copy for different params
class QQPTask(PairClassificationTask):
    ''' Task class for Quora Question Pairs. '''

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]
        self.scorer2 = F1Measure(1)
        self.val_metric = "%s_acc_f1" % name
        self.val_metric_decreases = False

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=4, label_idx=5, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=4, label_idx=5, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, has_labels=False, return_indices=True, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading QQP data.")

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        pcs, rcl, f1 = self.scorer2.get_metric(reset)
        return {'acc_f1': (acc + f1) / 2, 'accuracy': acc, 'f1': f1,
                'precision': pcs, 'recall': rcl}


@register_task('mnli-fiction', rel_path='MNLI/', genre='fiction')
@register_task('mnli-slate', rel_path='MNLI/', genre='slate')
@register_task('mnli-government', rel_path='MNLI/', genre='government')
@register_task('mnli-telephone', rel_path='MNLI/', genre='telephone')
@register_task('mnli-travel', rel_path='MNLI/', genre='travel')
class MultiNLISingleGenreTask(PairClassificationTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, genre, name, **kw):
        '''MNLI'''
        super(MultiNLISingleGenreTask, self).__init__(name, n_classes=3,
                                                      **kw)
        self.load_data(path, max_seq_len, genre)
        self.scorer2 = None
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, genre):
        '''Process the dataset located at path. We only use the in-genre matche data.'''
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(self._tokenizer_name,
            os.path.join(
                path,
                'train.tsv'),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=11,
            label_fn=targ_map.__getitem__,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=genre)

        val_matched_data = load_tsv(self._tokenizer_name,
            os.path.join(
                path,
                'dev_matched.tsv'),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=11,
            label_fn=targ_map.__getitem__,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=genre)

        te_matched_data = load_tsv(self._tokenizer_name,
            os.path.join(
                path,
                'test_matched.tsv'),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=genre)

        self.train_data_text = tr_data
        self.val_data_text = val_matched_data
        self.test_data_text = te_matched_data
        log.info("\tFinished loading MNLI " + genre + " data.")

    def get_metrics(self, reset=False):
        ''' No F1 '''
        return {'accuracy': self.scorer1.get_metric(reset)}


@register_task('mrpc', rel_path='MRPC/')
class MRPCTask(PairClassificationTask):
    ''' Task class for Microsoft Research Paraphase Task.  '''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' '''
        super(MRPCTask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]
        self.scorer2 = F1Measure(1)
        self.val_metric = "%s_acc_f1" % name
        self.val_metric_decreases = False

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=4, label_idx=0, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=4, label_idx=0, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=3, s2_idx=4, has_labels=False, return_indices=True, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading MRPC data.")

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        pcs, rcl, f1 = self.scorer2.get_metric(reset)
        return {'acc_f1': (acc + f1) / 2, 'accuracy': acc, 'f1': f1,
                'precision': pcs, 'recall': rcl}


@register_task('sts-b', rel_path='STS-B/')
@register_task('sts-b-alt', rel_path='STS-B/')  # second copy for different params
class STSBTask(PairRegressionTask):
    ''' Task class for Sentence Textual Similarity Benchmark.  '''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' '''
        super(STSBTask, self).__init__(name, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]
        #self.scorer1 = Average()
        #self.scorer2 = Average()
        self.scorer1 = Correlation("pearson")
        self.scorer2 = Correlation("spearman")
        self.val_metric = "%s_corr" % self.name
        self.val_metric_decreases = False

    def load_data(self, path, max_seq_len):
        ''' Load data '''
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, 'train.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=7, s2_idx=8, label_idx=9, label_fn=lambda x: float(x) / 5)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, 'dev.tsv'), max_seq_len, skip_rows=1,
                            s1_idx=7, s2_idx=8, label_idx=9, label_fn=lambda x: float(x) / 5)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=7, s2_idx=8, has_labels=False, return_indices=True, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading STS Benchmark data.")

    def get_metrics(self, reset=False):
        pearsonr = self.scorer1.get_metric(reset)
        spearmanr = self.scorer2.get_metric(reset)
        return {'corr': (pearsonr + spearmanr) / 2,
                'pearsonr': pearsonr, 'spearmanr': spearmanr}


@register_task('snli', rel_path='SNLI/')
class SNLITask(PairClassificationTask):
    ''' Task class for Stanford Natural Language Inference '''
    def __init__(self, path, max_seq_len, name, **kw):
        ''' Do stuff '''
        super(SNLITask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "train.tsv"), max_seq_len, label_fn=targ_map.__getitem__,
                           s1_idx=7, s2_idx=8, label_idx=10, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "dev.tsv"), max_seq_len, label_fn=targ_map.__getitem__,
                           s1_idx=7, s2_idx=8, label_idx=10, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=7, s2_idx=8, has_labels=False, return_indices=True,  skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SNLI data.")

@register_task('mnli', rel_path='MNLI/')
@register_task('mnli-alt', rel_path='MNLI/')  # second copy for different params
class MultiNLITask(PairClassificationTask):
    ''' Task class for Multi-Genre Natural Language Inference '''

    def __init__(self, path, max_seq_len, name, **kw):
        '''MNLI'''
        super(MultiNLITask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at path.'''
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=8, s2_idx=9, label_idx=11, label_fn=targ_map.__getitem__, skip_rows=1)

        # Warning to anyone who edits this: The reference label is column *15*, not 11 as above.
        val_matched_data = load_tsv(self._tokenizer_name, os.path.join(path, 'dev_matched.tsv'), max_seq_len,
                                    s1_idx=8, s2_idx=9, label_idx=15, label_fn=targ_map.__getitem__, skip_rows=1)
        val_mismatched_data = load_tsv(self._tokenizer_name, os.path.join(path, 'dev_mismatched.tsv'), max_seq_len,
                                       s1_idx=8, s2_idx=9, label_idx=15, label_fn=targ_map.__getitem__,
                                       skip_rows=1)
        val_data = [m + mm for m, mm in zip(val_matched_data, val_mismatched_data)]
        val_data = tuple(val_data)

        te_matched_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test_matched.tsv'), max_seq_len,
                                   s1_idx=8, s2_idx=9, targ_idx=None, idx_idx=0, skip_rows=1)
        te_mismatched_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test_mismatched.tsv'), max_seq_len,
                                      s1_idx=8, s2_idx=9, targ_idx=None, idx_idx=0, skip_rows=1)
        te_diagnostic_data = load_tsv(self._tokenizer_name, os.path.join(path, 'diagnostic.tsv'), max_seq_len,
                                      s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        te_data = [m + mm + d for m, mm, d in
                   zip(te_matched_data, te_mismatched_data, te_diagnostic_data)]

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading MNLI data.")


@register_task('mnli-diagnostic', rel_path='MNLI/')
class MultiNLIDiagnosticTask(PairClassificationTask):
    ''' Task class for diagnostic on MNLI'''

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=3, **kw)
        self.load_data_and_create_scorers(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data_and_create_scorers(self, path, max_seq_len):
        '''load MNLI diagnostics data. The tags for every column are loaded as indices.
        They will be converted to bools in preprocess_split function'''

        # Will create separate scorer for every tag. tag_group is the name of the
        # column it will have its own scorer
        def create_score_function(scorer, arg_to_scorer, tags_dict, tag_group):
            setattr(self, 'scorer__%s' % tag_group, scorer(arg_to_scorer))
            for index, tag in tags_dict.items():
                # 0 is missing value
                if index == 0:
                    continue
                setattr(self, "scorer__%s__%s" % (tag_group, tag), scorer(arg_to_scorer))

        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        diag_data_dic = load_diagnostic_tsv(self._tokenizer_name,
            os.path.join(
                path,
                'diagnostic-full.tsv'),
            max_seq_len,
            s1_col="Premise",
            s2_col="Hypothesis",
            label_col="Label",
            label_fn=targ_map.__getitem__,
            skip_rows=1)

        self.ix_to_lex_sem_dic = diag_data_dic['ix_to_lex_sem_dic']
        self.ix_to_pr_ar_str_dic = diag_data_dic['ix_to_pr_ar_str_dic']
        self.ix_to_logic_dic = diag_data_dic['ix_to_logic_dic']
        self.ix_to_knowledge_dic = diag_data_dic['ix_to_knowledge_dic']

        # Train, val, test splits are same. We only need one split but the code
        # probably expects all splits to be present.
        self.train_data_text = (
            diag_data_dic['sents1'],
            diag_data_dic['sents2'],
            diag_data_dic['targs'],
            diag_data_dic['idxs'],
            diag_data_dic['lex_sem'],
            diag_data_dic['pr_ar_str'],
            diag_data_dic['logic'],
            diag_data_dic['knowledge'])
        self.val_data_text = self.train_data_text
        self.test_data_text = self.train_data_text
        log.info("\tFinished loading MNLI Diagnostics data.")

        # TODO: use FastMatthews instead to save memory.
        create_score_function(Correlation, "matthews", self.ix_to_lex_sem_dic, 'lex_sem')
        create_score_function(Correlation, "matthews", self.ix_to_pr_ar_str_dic, 'pr_ar_str')
        create_score_function(Correlation, "matthews", self.ix_to_logic_dic, 'logic')
        create_score_function(Correlation, "matthews", self.ix_to_knowledge_dic, 'knowledge')
        log.info("\tFinished creating Score functions for Diagnostics data.")

    def update_diagnostic_metrics(self, logits, labels, batch):
        # Updates scorer for every tag in a given column (tag_group) and also the
        # the scorer for the column itself.
        def update_scores_for_tag_group(ix_to_tags_dic, tag_group):
            for ix, tag in ix_to_tags_dic.items():
                # 0 is for missing tag so here we use it to update scorer for the column
                # itself (tag_group).
                if ix == 0:
                    # This will contain 1s on positions where at least one of the tags of this
                    # column is present.
                    mask = batch[tag_group]
                    scorer_str = "scorer__%s" % tag_group
                # This branch will update scorers of individual tags in the column
                else:
                    # batch contains_field for every tag. It's either 0 or 1.
                    mask = batch["%s__%s" % (tag_group, tag)]
                    scorer_str = "scorer__%s__%s" % (tag_group, tag)

                # This will take only values for which the tag is true.
                indices_to_pull = torch.nonzero(mask)
                # No example in the batch is labeled with the tag.
                if indices_to_pull.size()[0] == 0:
                    continue
                sub_labels = labels[indices_to_pull[:, 0]]
                sub_logits = logits[indices_to_pull[:, 0]]
                scorer = getattr(self, scorer_str)
                scorer(sub_logits, sub_labels)
            return

        # Updates scorers for each tag.
        update_scores_for_tag_group(self.ix_to_lex_sem_dic, 'lex_sem')
        update_scores_for_tag_group(self.ix_to_pr_ar_str_dic, 'pr_ar_str')
        update_scores_for_tag_group(self.ix_to_logic_dic, 'logic')
        update_scores_for_tag_group(self.ix_to_knowledge_dic, 'knowledge')

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''

        def create_labels_from_tags(fields_dict, ix_to_tag_dict, tag_arr, tag_group):
            # If there is something in this row then tag_group should be set to 1.
            is_tag_group = 1 if len(tag_arr) != 0 else 0
            fields_dict[tag_group] = LabelField(is_tag_group, label_namespace=tag_group,
                                                skip_indexing=True)
            # For every possible tag in the column set 1 if the tag is present for
            # this example, 0 otherwise.
            for ix, tag in ix_to_tag_dict.items():
                if ix == 0:
                    continue
                is_present = 1 if ix in tag_arr else 0
                fields_dict['%s__%s' % (tag_group, tag)] = LabelField(
                    is_present, label_namespace='%s__%s' % (tag_group, tag), skip_indexing=True)
            return

        def _make_instance(input1, input2, label, idx, lex_sem, pr_ar_str, logic, knowledge):
            ''' from multiple types in one column create multiple fields '''
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["input2"] = sentence_to_text_field(input2, indexers)
            d["labels"] = LabelField(label, label_namespace="labels",
                                     skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idx",
                                  skip_indexing=True)
            d['sent1_str'] = MetadataField(" ".join(input1[1:-1]))
            d['sent2_str'] = MetadataField(" ".join(input2[1:-1]))

            # adds keys to dict "d" for every possible type in the column
            create_labels_from_tags(d, self.ix_to_lex_sem_dic, lex_sem, 'lex_sem')
            create_labels_from_tags(d, self.ix_to_pr_ar_str_dic, pr_ar_str, 'pr_ar_str')
            create_labels_from_tags(d, self.ix_to_logic_dic, logic, 'logic')
            create_labels_from_tags(d, self.ix_to_knowledge_dic, knowledge, 'knowledge')

            return Instance(d)

        instances = map(_make_instance, *split)
        #  return list(instances)
        return instances  # lazy iterator

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        collected_metrics = {}
        # We do not compute accuracy for this dataset but the eval function requires this key.
        collected_metrics["accuracy"] = 0

        def collect_metrics(ix_to_tag_dict, tag_group):
            for index, tag in ix_to_tag_dict.items():
                # Index 0 is used for missing data, here it will be used for score of the
                # whole category.
                if index == 0:
                    scorer_str = 'scorer__%s' % tag_group
                    scorer = getattr(self, scorer_str)
                    collected_metrics['%s' % (tag_group)] = scorer.get_metric(reset)
                else:
                    scorer_str = 'scorer__%s__%s' % (tag_group, tag)
                    scorer = getattr(self, scorer_str)
                    collected_metrics['%s__%s' % (tag_group, tag)] = scorer.get_metric(reset)

        collect_metrics(self.ix_to_lex_sem_dic, 'lex_sem')
        collect_metrics(self.ix_to_pr_ar_str_dic, 'pr_ar_str')
        collect_metrics(self.ix_to_logic_dic, 'logic')
        collect_metrics(self.ix_to_knowledge_dic, 'knowledge')
        return collected_metrics


@register_task('nps', rel_path='nps/')
class NPSTask(PairClassificationTask):

    def __init__(self, path, max_seq_len, name, probe_path="probe_dummy.tsv",
                 **kw):
        super(NPSTask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len, probe_path)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, probe_path):
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, 'train_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, has_labels=False, targ_map=targ_map, skip_rows=0)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, 'dev.tsv'), max_seq_len,
                            s1_idx=0, s2_idx=1, label_idx=2, targ_map=targ_map, skip_rows=0)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, has_labels=False, targ_map=targ_map, skip_rows=0)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading NP/S data.")


@register_task('rte', rel_path='RTE/')
class RTETask(PairClassificationTask):
    ''' Task class for Recognizing Textual Entailment 1, 2, 3, 5 '''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' '''
        super(RTETask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        ''' Process the datasets located at path. '''
        targ_map = {"not_entailment": 0, "entailment": 1}
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, 'train.tsv'), max_seq_len, targ_map=targ_map,
                           s1_idx=1, s2_idx=2, label_idx=3, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, 'dev.tsv'), max_seq_len, targ_map=targ_map,
                            s1_idx=1, s2_idx=2, label_idx=3, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, has_labels=False, return_indices=True, skip_rows=1)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading RTE.")


@register_task('qnli', rel_path='QNLI/')
@register_task('qnli-alt', rel_path='QNLI/')  # second copy for different params
class QNLITask(PairClassificationTask):
    '''Task class for SQuAD NLI'''

    def __init__(self, path, max_seq_len, name, **kw):
        super(QNLITask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        targ_map = {'not_entailment': 0, 'entailment': 1}
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "train.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=1, s2_idx=2, label_idx=3, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "dev.tsv"), max_seq_len, targ_map=targ_map,
                            s1_idx=1, s2_idx=2, label_idx=3, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, has_labels=False, return_indices=True, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading QNLI.")


@register_task('wnli', rel_path='WNLI/')
class WNLITask(PairClassificationTask):
    '''Class for Winograd NLI task'''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' '''
        super(WNLITask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=3, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=1, s2_idx=2, label_idx=3, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, has_labels=False, return_indices=True, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading Winograd.")


@register_task('joci', rel_path='JOCI/')
class JOCITask(PairOrdinalRegressionTask):
    '''Class for JOCI ordinal regression task'''

    def __init__(self, path, max_seq_len, name, **kw):
        super(JOCITask, self).__init__(name, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, 'train.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=0, s2_idx=1, label_idx=2)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, 'dev.tsv'), max_seq_len, skip_rows=1,
                            s1_idx=0, s2_idx=1, label_idx=2)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=0, s2_idx=1, label_idx=2)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading JOCI data.")


@register_task('wiki103_classif', rel_path='WikiText103/')
class Wiki103Classification(PairClassificationTask):
    '''Pair Classificaiton Task using Wiki103'''

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
        self.scorer2 = None
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.files_by_split = {'train': os.path.join(path, "train.sentences.txt"),
                               'val': os.path.join(path, "valid.sentences.txt"),
                               'test': os.path.join(path, "test.sentences.txt")}
        self.max_seq_len = max_seq_len
        self.min_seq_len = 0

    def get_split_text(self, split: str):
        ''' Get split text as iterable of records.
        Split should be one of 'train', 'val', or 'test'.
        '''
        return self.load_data(self.files_by_split[split])

    def load_data(self, path):
        ''' Rather than return a whole list of examples, stream them
        See WikiTextLMTask for an explanation of the preproc'''
        nonatomics_toks = [UNK_TOK_ALLENNLP, '<unk>']
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                sent = atomic_tokenize(toks, UNK_TOK_ATOMIC, nonatomics_toks, self.max_seq_len)
                if sent.count("=") >= 2 or len(toks) < self.min_seq_len + 2:
                    continue
                yield sent

    def get_sentences(self) -> Iterable[Sequence[str]]:
        ''' Yield sentences, used to compute vocabulary. '''
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for sent in self.load_data(path):
                yield sent

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process a language modeling split.  Split is a single list of sentences here.  '''
        def _make_instance(input1, input2, labels):
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["input2"] = sentence_to_text_field(input2, indexers)
            d["labels"] = LabelField(labels, label_namespace="labels",
                                     skip_indexing=True)
            return Instance(d)
        first = True
        for sent in split:
            if first:
                prev_sent = sent
                first = False
                continue
            yield _make_instance(prev_sent, sent, 1)
            prev_sent = sent

    def count_examples(self):
        ''' Compute here b/c we're streaming the sentences. '''
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            # pair sentence # = sent # - 1
            example_counts[split] = sum(1 for line in open(split_path)) - 1
        self.example_counts = example_counts


# Task class for DisSent with Wikitext 103 only considering clauses from within a single sentence
# Data sets should be prepared as described in Nie, Bennett, and Goodman (2017)
@register_task('dissentwiki', rel_path='DisSent/wikitext/',
               prefix="wikitext.dissent.single_sent")
# Task class for DisSent with Wikitext 103 considering clauses from within a single sentence
# or across two sentences.
# Data sets should be prepared as described in Nie, Bennett, and Goodman (2017)
@register_task('dissentwikifullbig', rel_path='DisSent/wikitext/',
               prefix="wikitext.dissent.big")
class DisSentTask(PairClassificationTask):
    ''' Task class for DisSent, dataset agnostic.
        Based on Nie, Bennett, and Goodman (2017), but with different datasets.
    '''

    def __init__(self, path, max_seq_len, prefix, name, **kw):
        ''' There are 8 classes because there are 8 discourse markers in
            the dataset (and, but, because, if, when, before, though, so)
        '''
        super().__init__(name, n_classes=8, **kw)
        self.max_seq_len = max_seq_len
        self.files_by_split = {"train": os.path.join(path, "%s.train" % prefix),
                               "val": os.path.join(path, "%s.valid" % prefix),
                               "test": os.path.join(path, "%s.test" % prefix)}

    def get_split_text(self, split: str):
        ''' Get split text as iterable of records.

        Split should be one of 'train', 'val', or 'test'.
        '''
        return self.load_data(self.files_by_split[split])

    def load_data(self, path):
        ''' Load data '''
        with open(path, 'r') as txt_fh:
            for row in txt_fh:
                row = row.strip().split('\t')
                if len(row) != 3 or not (row[0] and row[1] and row[2]):
                    continue
                sent1 = process_sentence(row[0], self.max_seq_len)
                sent2 = process_sentence(row[1], self.max_seq_len)
                targ = int(row[2])
                yield (sent1, sent2, targ)

    def get_sentences(self) -> Iterable[Sequence[str]]:
        ''' Yield sentences, used to compute vocabulary. '''
        for split in self.files_by_split:
            ''' Don't use test set for vocab building. '''
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for sent1, sent2, _ in self.load_data(path):
                yield sent1
                yield sent2

    def count_examples(self):
        ''' Compute the counts here b/c we're streaming the sentences. '''
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for line in open(split_path))
        self.example_counts = example_counts

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        def _make_instance(input1, input2, labels):
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["input2"] = sentence_to_text_field(input2, indexers)
            d["labels"] = LabelField(labels, label_namespace="labels",
                                     skip_indexing=True)
            return Instance(d)

        for sent1, sent2, trg in split:
            yield _make_instance(sent1, sent2, trg)


# TODO: does this even work? What is n_classes for this?
@register_task('weakgrounded', rel_path='mscoco/weakgrounded/')
class WeakGroundedTask(PairClassificationTask):
    ''' Task class for Weak Grounded Sentences i.e., training on pairs of captions for the same image '''

    def __init__(self, path, max_seq_len, n_classes, name, **kw):
        ''' Do stuff '''
        super(WeakGroundedTask, self).__init__(name, n_classes, **kw)

        ''' Process the dataset located at path.  '''
        ''' positive = captions of the same image, negative = captions of different images '''
        targ_map = {'negative': 0, 'positive': 1}
        targ_map = {'0': 0, '1': 1}

        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "train_aug.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=0, s2_idx=1, label_idx=2, skip_rows=0)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "val.tsv"), max_seq_len, targ_map=targ_map,
                            s1_idx=0, s2_idx=1, label_idx=2, skip_rows=0)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, "test.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=0, s2_idx=1, label_idx=2, skip_rows=0)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.n_classes = 2
        log.info("\tFinished loading MSCOCO data.")


@register_task('grounded', rel_path='mscoco/grounded/')
class GroundedTask(Task):
    ''' Task class for Grounded Sentences i.e., training on caption->image pair '''
    ''' Defined new metric function from AllenNLP Average '''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' Do stuff '''
        super(GroundedTask, self).__init__(name, **kw)
        self.scorer1 = Average()
        self.scorer2 = None
        self.val_metric = "%s_metric" % self.name
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + \
            self.val_data_text[0]
        self.ids = self.train_data_text[1] + \
            self.val_data_text[1]
        self.path = path
        self.img_encoder = None
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        metric = self.scorer1.get_metric(reset)

        return {'metric': metric}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        '''
        Convert a dataset of sentences into padded sequences of indices.
        Args:
            - split (list[list[str]]): list of inputs (possibly pair) and outputs
            - pair_input (int)
            - tok2idx (dict)
        Returns:
        '''
        def _make_instance(sent, label, ids):
            input1 = sentence_to_text_field(sent, indexers)
            label = NumericField(label)
            ids = NumericField(ids)
            return Instance({"input1": input1, "labels": label, "ids": ids})

        # Map over columns: input1, labels, ids
        instances = map(_make_instance, *split)
        #  return list(instances)
        return instances  # lazy iterator

    def load_data(self, path, max_seq_len):
        ''' Map sentences to image ids
            Keep track of caption ids just in case '''

        train, val, test = ([], [], []), ([], [], []), ([], [], [])

        with open(os.path.join(path, "train_idx.txt"), 'r') as f:
            train_ids = [item.strip() for item in f.readlines()]
        with open(os.path.join(path, "val_idx.txt"), 'r') as f:
            val_ids = [item.strip() for item in f.readlines()]
        with open(os.path.join(path, "test_idx.txt"), 'r') as f:
            test_ids = [item.strip() for item in f.readlines()]

        f = open(os.path.join(path, "train.json"), 'r')
        for line in f:
            tr_dict = json.loads(line)
        f = open(os.path.join(path, "val.json"), 'r')
        for line in f:
            val_dict = json.loads(line)
        f = open(os.path.join(path, "test.json"), 'r')
        for line in f:
            te_dict = json.loads(line)
        with open(os.path.join(path, "feat_map.json")) as fd:
            keymap = json.load(fd)

        def load_mscoco(data_dict, data_list, img_idxs):
            for img_idx in img_idxs:
                newimg_id = 'mscoco/grounded/' + img_idx + '.json'
                for caption_id in data_dict[img_idx]['captions']:
                    data_list[0].append(data_dict[img_idx]['captions'][caption_id])
                    data_list[1].append(1)
                    data_list[2].append(int(keymap[newimg_id]))
            return data_list

        train = load_mscoco(tr_dict, train, train_ids)
        val = load_mscoco(val_dict, val, val_ids)
        test = load_mscoco(te_dict, test, test_ids)

        self.tr_data = train
        self.val_data = val
        self.te_data = test
        self.train_data_text = train
        self.val_data_text = val
        self.test_data_text = test

        log.info("\tTrain: %d, Val: %d, Test: %d", len(train[0]), len(val[0]), len(test[0]))
        log.info("\tFinished loading MSCOCO data!")


@register_task('groundedsw', rel_path='mscoco/grounded')
class GroundedSWTask(Task):
    ''' Task class for Grounded Sentences i.e., training on caption->image pair '''
    ''' Defined new metric function from AllenNLP Average '''

    def __init__(self, path, max_seq_len, name, **kw):
        super(GroundedSWTask, self).__init__(name, **kw)
        self.scorer1 = Average()
        self.scorer2 = None
        self.val_metric = "%s_metric" % self.name
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + \
            self.val_data_text[0]
        self.ids = self.train_data_text[1] + \
            self.val_data_text[1]
        self.path = path
        self.img_encoder = None
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        metric = self.scorer1.get_metric(reset)

        return {'metric': metric}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        '''
        Convert a dataset of sentences into padded sequences of indices.
        Args:
            - split (list[list[str]]): list of inputs (possibly pair) and outputs
            - pair_input (int)
            - tok2idx (dict)
        Returns:
        '''
        def _make_instance(sent, label, ids):
            input1 = sentence_to_text_field(sent, indexers)
            label = NumericField(label)
            ids = NumericField(ids)
            return Instance({"input1": input1, "labels": label, "ids": ids})

        # Map over columns: input1, labels, ids
        instances = map(_make_instance, *split)
        #  return list(instances)
        return instances  # lazy iterator

    def load_data(self, path, max_seq_len):
        ''' Map sentences to image ids
            Keep track of caption ids just in case '''

        train, val, test = ([], [], []), ([], [], []), ([], [], [])

        def get_data(dataset, data):
            f = open(path + dataset + ".tsv", 'r')
            for line in f:
                items = line.strip().split('\t')
                if len(items) < 3 or items[1] == '0':
                    continue
                data[0].append(items[0])
                data[1].append(int(items[1]))
                data[2].append(int(items[2]))
            return data

        train = get_data('shapeworld/train', train)
        val = get_data('shapeworld/val', val)
        test = get_data('shapeworld/test', test)

        self.tr_data = train
        self.val_data = val
        self.te_data = test
        self.train_data_text = train
        self.val_data_text = val
        self.test_data_text = test

        log.info("Train: %d, Val: %d, Test: %d", len(train[0]), len(val[0]), len(test[0]))
        log.info("\nFinished loading SW data!")


@register_task('recast-puns', rel_path='DNC/recast_puns_data')
@register_task('recast-ner', rel_path='DNC/recast_ner_data')
@register_task('recast-verbnet', rel_path='DNC/recast_verbnet_data')
@register_task('recast-verbcorner', rel_path='DNC/recast_verbcorner_data')
@register_task('recast-sentiment', rel_path='DNC/recast_sentiment_data')
@register_task('recast-factuality', rel_path='DNC/recast_factuality_data')
@register_task('recast-winogender', rel_path='DNC/manually-recast-winogender')
@register_task('recast-lexicosyntax', rel_path='DNC/lexicosyntactic_recasted')
@register_task('recast-kg', rel_path='DNC/kg-relations')
class RecastNLITask(PairClassificationTask):
    ''' Task class for NLI Recast Data'''

    def __init__(self, path, max_seq_len, name, **kw):
        super(RecastNLITask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, skip_rows=0, label_idx=3)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, 'dev.tsv'), max_seq_len,
                            s1_idx=0, s2_idx=1, skip_rows=0, label_idx=3)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, skip_rows=0, label_idx=3)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading recast probing data.")


class TaggingTask(Task):
    ''' Generic tagging task, one tag per word '''

    def __init__(self, name, num_tags, **kw):
        super().__init__(name, **kw)
        assert num_tags > 0
        self.num_tags = num_tags + 2  # add tags for unknown and padding
        self.scorer1 = CategoricalAccuracy()
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.all_labels = [str(i) for i in range(self.num_tags)]
        self._label_namespace = self.name + "_tags"
        self.target_indexer = {"words": SingleIdTokenIndexer(namespace=self._label_namespace)}

    def truncate(self, max_seq_len,
                 sos_tok=utils.SOS_TOK, eos_tok=utils.EOS_TOK):
        ''' Truncate the data if any sentences are longer than max_seq_len. '''
        self.train_data_text = [truncate(self.train_data_text[0], max_seq_len,
                                         sos_tok, eos_tok), self.train_data_text[1]]
        self.val_data_text = [truncate(self.val_data_text[0], max_seq_len,
                                       sos_tok, eos_tok), self.val_data_text[1]]
        self.test_data_text = [truncate(self.test_data_text[0], max_seq_len,
                                        sos_tok, eos_tok), self.test_data_text[1]]

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        return {'accuracy': acc}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process a tagging task '''
        inputs = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        targs = [TextField(list(map(Token, sent)), token_indexers=self.target_indexer)
                 for sent in split[2]]
        # Might be better as LabelField? I don't know what these things mean
        instances = [Instance({"inputs": x, "targs": t}) for (x, t) in zip(inputs, targs)]
        return instances

    def get_all_labels(self) -> List[str]:
        return self.all_labels


@register_task('ccg', rel_path='CCG/')
class CCGTaggingTask(TaggingTask):
    ''' CCG supertagging as a task.
        Using the supertags from CCGbank. '''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' There are 1363 supertags in CCGBank. '''
        super().__init__(name, num_tags=1363, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at each data file.
           The target needs to be split into tokens because
           it is a sequence (one tag per input token). '''
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, "ccg_1363.train"), max_seq_len,
                           s1_idx=0, s2_idx=None, label_idx=1, label_fn=lambda t: t.split(' '))
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, "ccg_1363.dev"), max_seq_len,
                            s1_idx=0, s2_idx=None, label_idx=1, label_fn=lambda t: t.split(' '))
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'ccg_1363.test'), max_seq_len,
                           s1_idx=0, s2_idx=None, label_idx=1, label_fn=lambda t: t.split(' '))
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info('\tFinished loading CCGTagging data.')
