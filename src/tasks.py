'''Define the tasks and code for loading their data.

- As much as possible, following the existing task hierarchy structure.
- When inheriting, be sure to write and call load_data.
- Set all text data as an attribute, task.sentences (List[List[str]])
- Each task's val_metric should be name_metric, where metric is returned by get_metrics()
'''
import copy
import collections
import itertools
import functools
import os
import math
import logging as log
import json
import numpy as np
from typing import Iterable, Sequence, Any, Type

from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Average
from allennlp.data.token_indexers import SingleIdTokenIndexer
from .allennlp_mods.correlation import Correlation

# Fields for instance processing
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, LabelField
from .allennlp_mods.numeric_field import NumericField

from . import serialize
from .utils import load_tsv, process_sentence, truncate

def _sentence_to_text_field(sent: Sequence[str], indexers: Any):
    return TextField(list(map(Token, sent)), token_indexers=indexers)


def process_single_pair_task_split(split, indexers, is_pair=True, classification=True):
    '''
    Convert a dataset of sentences into padded sequences of indices. Shared
    across several classes.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)

    Returns:
    '''
    def _make_instance(input1, input2, labels, idx=None):
        d = {}
        d["input1"] = _sentence_to_text_field(input1, indexers)
        if input2:
            d["input2"] = _sentence_to_text_field(input2, indexers)
        if classification:
            d["labels"] = LabelField(labels, label_namespace="labels",
                                     skip_indexing=True)
        else:
            d["labels"] = NumericField(labels)

        if idx is not None:  # numbered test examples
            d["idx"] = LabelField(idx, label_namespace="idxs",
                                  skip_indexing=True)
        return Instance(d)

    if not is_pair:  # dummy iterator for input2
        split = list(split)
        split[1] = itertools.repeat(None)
    # Map over columns: input2, (input2), labels, (idx)
    instances = map(_make_instance, *split)
    #  return list(instances)
    return instances  # lazy iterator


class Task():
    '''Generic class for a task

    Methods and attributes:
        - load_data: load dataset from a path and create splits
        - truncate: truncate data to be at most some length
        - get_metrics:

    Outside the task:
        - process: pad and indexify data given a mapping
        - optimizer
    '''

    def __init__(self, name):
        self.name = name

    def load_data(self, path, max_seq_len):
        ''' Load data from path and create splits. '''
        raise NotImplementedError

    def truncate(self, max_seq_len, sos_tok, eos_tok):
        ''' Shorten sentences to max_seq_len and add sos and eos tokens. '''
        raise NotImplementedError

    def get_sentences(self) -> Iterable[Sequence[str]]:
        ''' Yield sentences, used to compute vocabulary. '''
        yield from self.sentences

    def get_split_text(self, split: str):
        ''' Get split text, typically as list of columns.

        Split should be one of 'train', 'val', or 'test'.
        '''
        return getattr(self, '%s_data_text' % split)

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        raise NotImplementedError

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        raise NotImplementedError


class ClassificationTask(Task):
    ''' General classification task '''
    def __init__(self, name):
        super().__init__(name)


class RegressionTask(Task):
    ''' General regression task '''
    def __init__(self, name):
        super().__init__(name)


class SingleClassificationTask(ClassificationTask):
    ''' Generic sentence pair classification '''

    def __init__(self, name, n_classes):
        super().__init__(name)
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

    def __init__(self, name, n_classes):
        super().__init__(name)
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


class NLIProbingTask(PairClassificationTask):
    ''' Generic probing with NLI test data (cannot be used for train or eval)'''

    def __init__(self, name, n_classes):
        super().__init__(name)


class PairRegressionTask(RegressionTask):
    ''' Generic sentence pair classification '''

    def __init__(self, name):
        super().__init__(name)
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

    def __init__(self, name):
        super().__init__(name)
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

    def __init__(self, name):
        super().__init__(name)
        self.scorer1 = Average()  # for average BLEU or something
        self.scorer2 = None
        self.val_metric = "%s_bleu" % self.name
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        bleu = self.scorer1.get_metric(reset)
        return {'bleu': bleu}


class RankingTask(Task):
    ''' Generic sentence ranking task, given some input '''

    def __init__(self, name, n_choices):
        super().__init__(name)
        self.n_choices = n_choices


class LanguageModelingTask(SequenceGenerationTask):
    ''' Generic language modeling task '''

    def __init__(self, name):
        super().__init__(name)
        self.scorer1 = Average()
        self.scorer2 = None
        self.val_metric = "%s_perplexity" % self.name
        self.val_metric_decreases = True

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        nll = self.scorer1.get_metric(reset)
        return {'perplexity': math.exp(nll)}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process a language modeling split.

        Split is a single list of sentences here.
        '''
        inp_fwd = [TextField(list(map(Token, sent[:-1])), token_indexers=indexers) for sent in split]
        inp_bwd = [TextField(list(map(Token, sent[::-1][:-1])), token_indexers=indexers)
                   for sent in split]
        if "chars" not in indexers:
            targs_indexers = {"words": SingleIdTokenIndexer()}
        else:
            targs_indexers = indexers
        trg_fwd = [TextField(list(map(Token, sent[1:])), token_indexers=targs_indexers)
                   for sent in split]
        trg_bwd = [TextField(list(map(Token, sent[::-1][1:])), token_indexers=targs_indexers)
                   for sent in split]
        # instances = [Instance({"input": inp, "targs": trg_f, "targs_b": trg_b})
        #             for (inp, trg_f, trg_b) in zip(inputs, trg_fwd, trg_bwd)]
        instances = [Instance({"input": inp_f, "input_bwd": inp_b, "targs": trg_f, "targs_b": trg_b})
                     for (inp_f, inp_b, trg_f, trg_b) in zip(inp_fwd, inp_bwd, trg_fwd, trg_bwd)]
        #instances = [Instance({"input": inp_f, "targs": trg_f}) for (inp_f, trg_f) in zip(inp_fwd, trg_fwd)]
        return instances


class WikiTextLMTask(LanguageModelingTask):
    ''' Language modeling task on Wikitext '''

    def __init__(self, path, max_seq_len, name="wiki"):
        super().__init__(name)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text + self.val_data_text

    def load_data(self, path, max_seq_len):
        tr_data = self.load_txt(os.path.join(path, "train.txt"), max_seq_len)
        val_data = self.load_txt(os.path.join(path, "valid.txt"), max_seq_len)
        te_data = self.load_txt(os.path.join(path, "test.txt"), max_seq_len)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading WikiText")

    def load_txt(self, path, max_seq_len):
        data = []
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if toks == '':
                    continue
                data.append(process_sentence(toks, max_seq_len))
        return data


class WikiText2LMTask(WikiTextLMTask):
    ''' Language modeling task on Wikitext 2'''

    def __init__(self, path, max_seq_len, name="wiki2"):
        super().__init__(path, max_seq_len, name)


class WikiText103LMTask(WikiTextLMTask):
    ''' Language modeling task on Wikitext 103'''

    def __init__(self, path, max_seq_len, name="wiki103"):
        super().__init__(path, max_seq_len, name)


class BWBLMTask(WikiTextLMTask):
    ''' Language modeling task on Billion Word Benchmark'''

    def __init__(self, path, max_seq_len, name="bwb"):
        super().__init__(path, max_seq_len, name)


class SSTTask(SingleClassificationTask):
    ''' Task class for Stanford Sentiment Treebank.  '''

    def __init__(self, path, max_seq_len, name="sst"):
        ''' '''
        super(SSTTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]

    def load_data(self, path, max_seq_len):
        ''' Load data '''
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=0, s2_idx=None, targ_idx=1, skip_rows=1)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len,
                            s1_idx=0, s2_idx=None, targ_idx=1, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=None, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SST data.")


class RedditTask(RankingTask):
    ''' Task class for Reddit data.  '''

    def __init__(self, path, max_seq_len, name="reddit"):
        ''' '''
        super(RedditTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1]  + self.val_data_text[0] + self.val_data_text[1]
        #:pdb.set_trace()
        self.scorer1 = Average() #CategoricalAccuracy()
        self.scorer2 = None
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = True

    def load_data(self, path, max_seq_len):
        ''' Load data '''
        print("Loading data")
        print("LOADING REDDIT DATA FROM A DIFF LOCATION COMPARED TO REST OF THE TEAM. PLEASE CHANGE")
        path = '//nfs/jsalt/home/raghu/'
        tr_data = load_tsv(os.path.join(path, 'train_2008_Random.csv'), max_seq_len,
                           s1_idx=2, s2_idx=3, targ_idx=None, skip_rows=0)
        print("FINISHED LOADING TRAIN DATA")
        dev_data = load_tsv(os.path.join(path, 'dev_2008_Random.csv'), max_seq_len,
                           s1_idx=2, s2_idx=3, targ_idx=None, skip_rows=0)
        print("FINISHED LOADING dev DATA")
        test_data = load_tsv(os.path.join(path, 'dev_2008_Random.csv'), max_seq_len,
                           s1_idx=2, s2_idx=3, targ_idx=None, skip_rows=0)
        print("FINISHED LOADING test DATA")
        self.train_data_text = tr_data
        self.val_data_text = dev_data
        self.test_data_text = test_data
        log.info("\tFinished loading Temporary Reddit data.")

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        return process_single_pair_task_split(split, indexers, is_pair=True)

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        #pdb.set_trace()
        acc = self.scorer1.get_metric(reset)
        return {'accuracy': acc}


class CoLATask(SingleClassificationTask):
    '''Class for Warstdadt acceptability task'''

    def __init__(self, path, max_seq_len, name="acceptability"):
        ''' '''
        super(CoLATask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        #self.scorer1 = Average()
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=None, targ_idx=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=None, targ_idx=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=None, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading CoLA.")

    def get_metrics(self, reset=False):
        return {'mcc': self.scorer1.get_metric(reset),
                'accuracy': self.scorer2.get_metric(reset)}


class QQPTask(PairClassificationTask):
    ''' Task class for Quora Question Pairs. '''

    def __init__(self, path, max_seq_len, name="qqp"):
        super().__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]
        self.scorer2 = F1Measure(1)
        self.val_metric = "%s_acc_f1" % name
        self.val_metric_decreases = False

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
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


class MultiNLISingleGenreTask(PairClassificationTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, genre, name):
        '''MNLI'''
        super(MultiNLISingleGenreTask, self).__init__(name, 3)
        self.load_data(path, max_seq_len, genre)
        self.scorer2 = None
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, genre):
        '''Process the dataset located at path. We only use the in-genre matche data.'''
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

        tr_data = load_tsv(
            os.path.join(
                path,
                'train.tsv'),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            targ_idx=11,
            targ_map=targ_map,
            skip_rows=1,
            filter_idx=3,
            filter_value=genre)

        val_matched_data = load_tsv(
            os.path.join(
                path,
                'dev_matched.tsv'),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            targ_idx=11,
            targ_map=targ_map,
            skip_rows=1,
            filter_idx=3,
            filter_value=genre)

        te_matched_data = load_tsv(
            os.path.join(
                path,
                'test_matched.tsv'),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            targ_idx=None,
            idx_idx=0,
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


class MultiNLIFictionTask(MultiNLISingleGenreTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, name="mnli-fiction"):
        '''MNLI'''
        super(
            MultiNLIFictionTask,
            self).__init__(
            path,
            max_seq_len,
            genre="fiction",
            name=name)


class MultiNLISlateTask(MultiNLISingleGenreTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, name="mnli-slate"):
        '''MNLI'''
        super(MultiNLISlateTask, self).__init__(path, max_seq_len, genre="slate", name=name)


class MultiNLIGovernmentTask(MultiNLISingleGenreTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, name="mnli-government"):
        '''MNLI'''
        super(
            MultiNLIGovernmentTask,
            self).__init__(
            path,
            max_seq_len,
            genre="government",
            name=name)


class MultiNLITelephoneTask(MultiNLISingleGenreTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, name="mnli-telephone"):
        '''MNLI'''
        super(
            MultiNLITelephoneTask,
            self).__init__(
            path,
            max_seq_len,
            genre="telephone",
            name=name)


class MultiNLITravelTask(MultiNLISingleGenreTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, name="mnli-travel"):
        '''MNLI'''
        super(
            MultiNLITravelTask,
            self).__init__(
            path,
            max_seq_len,
            genre="travel",
            name=name)


class MRPCTask(PairClassificationTask):
    ''' Task class for Microsoft Research Paraphase Task.  '''

    def __init__(self, path, max_seq_len, name="mrpc"):
        ''' '''
        super(MRPCTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]
        self.scorer2 = F1Measure(1)
        self.val_metric = "%s_acc_f1" % name
        self.val_metric_decreases = False

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=0, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=4, targ_idx=0, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=None, idx_idx=0, skip_rows=1)
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


class STSBTask(PairRegressionTask):
    ''' Task class for Sentence Textual Similarity Benchmark.  '''

    def __init__(self, path, max_seq_len, name="sts_benchmark"):
        ''' '''
        super(STSBTask, self).__init__(name)
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
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: float(x) / 5)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len, skip_rows=1,
                            s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: float(x) / 5)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=7, s2_idx=8, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading STS Benchmark data.")

    def get_metrics(self, reset=False):
        pearsonr = self.scorer1.get_metric(reset)
        spearmanr = self.scorer2.get_metric(reset)
        return {'corr': (pearsonr + spearmanr) / 2,
                'pearsonr': pearsonr, 'spearmanr': spearmanr}


class SNLITask(PairClassificationTask):
    ''' Task class for Stanford Natural Language Inference '''

    def __init__(self, path, max_seq_len, name="snli"):
        ''' Do stuff '''
        super(SNLITask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=7, s2_idx=8, targ_idx=-1, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len, targ_map=targ_map,
                            s1_idx=7, s2_idx=8, targ_idx=-1, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=7, s2_idx=8, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SNLI data.")


class MultiNLITask(PairClassificationTask):
    ''' Task class for Multi-Genre Natural Language Inference '''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''
        super(MultiNLITask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at path.'''
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=8, s2_idx=9, targ_idx=11, targ_map=targ_map, skip_rows=1)
        val_matched_data = load_tsv(os.path.join(path, 'dev_matched.tsv'), max_seq_len,
                                    s1_idx=8, s2_idx=9, targ_idx=11, targ_map=targ_map, skip_rows=1)
        val_mismatched_data = load_tsv(os.path.join(path, 'dev_mismatched.tsv'), max_seq_len,
                                       s1_idx=8, s2_idx=9, targ_idx=11, targ_map=targ_map,
                                       skip_rows=1)
        val_data = [m + mm for m, mm in zip(val_matched_data, val_mismatched_data)]
        val_data = tuple(val_data)

        te_matched_data = load_tsv(os.path.join(path, 'test_matched.tsv'), max_seq_len,
                                   s1_idx=8, s2_idx=9, targ_idx=None, idx_idx=0, skip_rows=1)
        te_mismatched_data = load_tsv(os.path.join(path, 'test_mismatched.tsv'), max_seq_len,
                                      s1_idx=8, s2_idx=9, targ_idx=None, idx_idx=0, skip_rows=1)
        te_diagnostic_data = load_tsv(os.path.join(path, 'diagnostic.tsv'), max_seq_len,
                                      s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        te_data = [m + mm + d for m, mm, d in
                   zip(te_matched_data, te_mismatched_data, te_diagnostic_data)]
        te_data[3] = list(range(len(te_data[3])))

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading MNLI data.")

class NLITypeProbingTask(PairClassificationTask):
    ''' Task class for Probing Task (NLI-type)'''

    def __init__(self, path, max_seq_len, name="nli-prob"):
        super(NLITypeProbingTask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(os.path.join(path, 'train_dummy.tsv'), max_seq_len,
                        s1_idx=1, s2_idx=2, targ_idx=None, targ_map=targ_map, skip_rows=0)
        val_data = load_tsv(os.path.join(path, 'dat_cv/with.cvcv.mnli'), max_seq_len,
                        s1_idx=0, s2_idx=1, targ_idx=2, targ_map=targ_map, skip_rows=0)
        te_data = load_tsv(os.path.join(path, 'test_dummy.tsv'), max_seq_len,
                        s1_idx=1, s2_idx=2, targ_idx=None, targ_map=targ_map, skip_rows=0)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading NLI-type probing data.")


class MultiNLIAltTask(MultiNLITask):
    ''' Task class for Multi-Genre Natural Language Inference.

    Identical to MultiNLI class, but it can be handy to have two when controlling model settings.
    '''

    def __init__(self, path, max_seq_len, name="mnli-alt"):
        '''MNLI'''
        super(MultiNLIAltTask, self).__init__(path, max_seq_len, name)


class RTETask(PairClassificationTask):
    ''' Task class for Recognizing Textual Entailment 1, 2, 3, 5 '''

    def __init__(self, path, max_seq_len, name="rte"):
        ''' '''
        super(RTETask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        ''' Process the datasets located at path. '''
        targ_map = {"not_entailment": 0, "entailment": 1}
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len, targ_map=targ_map,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len, targ_map=targ_map,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading RTE.")


class QNLITask(PairClassificationTask):
    '''Task class for SQuAD NLI'''

    def __init__(self, path, max_seq_len, name="squad"):
        super(QNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        targ_map = {'not_entailment': 0, 'entailment': 1}
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len, targ_map=targ_map,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading QNLI.")


class WNLITask(PairClassificationTask):
    '''Class for Winograd NLI task'''

    def __init__(self, path, max_seq_len, name="winograd"):
        ''' '''
        super(WNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading Winograd.")


class JOCITask(PairOrdinalRegressionTask):
    '''Class for JOCI ordinal regression task'''

    def __init__(self, path, max_seq_len, name="joci"):
        super(JOCITask, self).__init__(name)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=0, s2_idx=1, targ_idx=2)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len, skip_rows=1,
                            s1_idx=0, s2_idx=1, targ_idx=2)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=0, s2_idx=1, targ_idx=2)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading JOCI data.")


class PDTBTask(PairClassificationTask):
    ''' Task class for discourse relation prediction using PDTB'''

    def __init__(self, path, max_seq_len, name="pdtb"):
        ''' Load data and initialize'''
        super(PDTBTask, self).__init__(name, 99)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''

        tr_data = load_tsv(os.path.join(path, "pdtb_sentence_pairs.train.txt"), max_seq_len,
                           s1_idx=4, s2_idx=5, targ_idx=3)
        val_data = load_tsv(os.path.join(path, "pdtb_sentence_pairs.dev.txt"), max_seq_len,
                            s1_idx=4, s2_idx=5, targ_idx=3)
        te_data = load_tsv(os.path.join(path, "pdtb_sentence_pairs.test.txt"), max_seq_len,
                           s1_idx=4, s2_idx=5, targ_idx=3)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading PDTB data.")


class MTTask(SequenceGenerationTask):
    '''Machine Translation Task'''

    def __init__(self, path, max_seq_len, name='MTTask'):
        super().__init__(name)
        self.scorer1 = Average()
        self.scorer2 = None
        self.val_metric = "%s_perplexity" % self.name
        self.val_metric_decreases = True
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0] + \
            self.train_data_text[2] + self.val_data_text[2]

    def load_data(self, path, max_seq_len):
        self.train_data_text = load_tsv(os.path.join(path, 'train.txt'), max_seq_len,
                                        s1_idx=0, s2_idx=None, targ_idx=1,
                                        targ_fn=lambda t: t.split(' '))
        self.val_data_text = load_tsv(os.path.join(path, 'valid.txt'), max_seq_len,
                                      s1_idx=0, s2_idx=None, targ_idx=1,
                                      targ_fn=lambda t: t.split(' '))
        self.test_data_text = load_tsv(os.path.join(path, 'test.txt'), max_seq_len,
                                       s1_idx=0, s2_idx=None, targ_idx=1,
                                       targ_fn=lambda t: t.split(' '))

        log.info("\tFinished loading MT data.")

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        ppl = self.scorer1.get_metric(reset)
        return {'perplexity': ppl}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process a machine translation split '''
        def _make_instance(input, target):
            d = {}
            d["inputs"] = _sentence_to_text_field(input, indexers)
            d["targs"] = _sentence_to_text_field(target, indexers)
            return Instance(d)
        # Map over columns: inputs, targs
        instances = map(_make_instance, split[0], split[2])
        #  return list(instances)
        return instances  # lazy iterator


class WikiInsertionsTask(MTTask):
    '''Task which predicts a span to insert at a given index'''

    def __init__(self, path, max_seq_len, name='WikiInsertionTask'):
        super().__init__(path, max_seq_len, name)
        self.scorer1 = Average()
        self.scorer2 = None
        self.val_metric = "%s_perplexity" % self.name
        self.val_metric_decreases = True
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.target_sentences = self.train_data_text[2] + self.val_data_text[2]

    def load_data(self, path, max_seq_len):
        self.train_data_text = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len,
                                        s1_idx=0, s2_idx=None, targ_idx=3, skip_rows=1,
                                        targ_fn=lambda t: t.split(' '))
        self.val_data_text = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len,
                                      s1_idx=0, s2_idx=None, targ_idx=3, skip_rows=1,
                                      targ_fn=lambda t: t.split(' '))
        self.test_data_text = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                                       s1_idx=0, s2_idx=None, targ_idx=3, skip_rows=1,
                                       targ_fn=lambda t: t.split(' '))
        log.info("\tFinished loading WikiInsertions data.")

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        ppl = self.scorer1.get_metric(reset)
        return {'perplexity': ppl}


class DisSentBWBSingleTask(PairClassificationTask):
    ''' Task class for DisSent with the Billion Word Benchmark'''

    def __init__(self, path, max_seq_len, name="dissentbwb"):
        super().__init__(name, 8)  # 8 classes, for 8 discource markers
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
        tr_data = load_tsv(os.path.join(path, "bwb.dissent.single_sent.train"), max_seq_len,
                           s1_idx=0, s2_idx=1, targ_idx=2)
        val_data = load_tsv(os.path.join(path, "bwb.dissent.single_sent.valid"), max_seq_len,
                            s1_idx=0, s2_idx=1, targ_idx=2)
        te_data = load_tsv(os.path.join(path, 'bwb.dissent.single_sent.test'), max_seq_len,
                           s1_idx=0, s2_idx=1, targ_idx=2)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading DisSent data.")


class DisSentWikiSingleTask(PairClassificationTask):
    ''' Task class for DisSent with Wikitext 103 only considering clauses from within a single sentence'''

    def __init__(self, path, max_seq_len, name="dissentwiki"):
        super().__init__(name, 8)  # 8 classes, for 8 discource markers
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
        tr_data = load_tsv(os.path.join(path, "wikitext.dissent.single_sent.train"), max_seq_len,
                           s1_idx=0, s2_idx=1, targ_idx=2)
        val_data = load_tsv(os.path.join(path, "wikitext.dissent.single_sent.valid"), max_seq_len,
                            s1_idx=0, s2_idx=1, targ_idx=2)
        te_data = load_tsv(os.path.join(path, 'wikitext.dissent.single_sent.test'), max_seq_len,
                           s1_idx=0, s2_idx=1, targ_idx=2)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading DisSent data.")


class DisSentWikiFullTask(PairClassificationTask):
    ''' Task class for DisSent with Wikitext 103 only considering clauses from within a single sentence'''

    def __init__(self, path, max_seq_len, name="dissentwikifull"):
        super().__init__(name, 8)  # 8 classes, for 8 discource markers
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
        tr_data = load_tsv(os.path.join(path, "wikitext.dissent.train"), max_seq_len,
                           s1_idx=0, s2_idx=1, targ_idx=2)
        val_data = load_tsv(os.path.join(path, "wikitext.dissent.valid"), max_seq_len,
                            s1_idx=0, s2_idx=1, targ_idx=2)
        te_data = load_tsv(os.path.join(path, 'wikitext.dissent.test'), max_seq_len,
                           s1_idx=0, s2_idx=1, targ_idx=2)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading DisSent data.")


class WeakGroundedTask(PairClassificationTask):
    ''' Task class for Weak Grounded Sentences i.e., training on pairs of captions for the same image '''

    def __init__(self, path, max_seq_len, n_classes, name="weakgrounded"):
        ''' Do stuff '''
        super(WeakGroundedTask, self).__init__(name, n_classes)

        ''' Process the dataset located at path.  '''
        ''' positive = captions of the same image, negative = captions of different images '''
        targ_map = {'negative': 0, 'positive': 1}
        targ_map = {'0': 0, '1': 1}

        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=0, s2_idx=1, targ_idx=2, skip_rows=0)
        val_data = load_tsv(os.path.join(path, "val.tsv"), max_seq_len, targ_map=targ_map,
                            s1_idx=0, s2_idx=1, targ_idx=2, skip_rows=0)
        te_data = load_tsv(os.path.join(path, "test.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=0, s2_idx=1, targ_idx=2, skip_rows=0)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.n_classes = 2
        log.info("\tFinished loading MSCOCO data.")


class GroundedTask(Task):
    ''' Task class for Grounded Sentences i.e., training on caption->image pair '''
    ''' Defined new metric function from AllenNLP Average '''
    ''' Specify metric name as 'cos_sim' or 'abs_diff' '''

    def __init__(self, path, max_seq_len, name="grounded"):
        ''' Do stuff '''
        super(GroundedTask, self).__init__(name)
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
        self.loss_fn = nn.CosineEmbeddingLoss()
        self.metric_fn = nn.PairwiseDistance(p=1, eps=1e-6)
        self.val_metric_decreases = True
        '''
        self.metric_fn = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.val_metric_decreases = False
        '''


    def _compute_metric(self, metric_name, tensor1, tensor2):
        '''Metrics for similarity in image space'''

        np1, np2 = tensor1.data.numpy(), tensor2.data.numpy()

        if metric_name is 'abs_diff':
            metric = np.mean(np1 - np2)
        elif metric_name is 'cos_sim':
            metric = cos_sim(np.asarray(np1), np.asarray(np2))[0][0]
        else:
            print('Undefined metric name!')
            metric = 0

        return metric
        
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
            input1 = _sentence_to_text_field(sent, indexers)
            label = NumericField(label)
            ids = NumericField(ids)
            return Instance({"input1": input1, "labels": label, "ids": ids})

        # Map over columns: input1, labels, ids
        instances = map(_make_instance, *split)
        #  return list(instances)
        return instances  # lazy iterator


    def load_data(self, path, max_seq_len):
        '''Map sentences to image ids (keep track of sentence ids just in case)'''

        # changed for temp
        train, val, test = ([], [], []), ([], [], []), ([], [], [])
        
        train_ids = [item for item in os.listdir(os.path.join(path, "train")) if '.DS' not in item]
        val_ids = [item for item in os.listdir(os.path.join(path, "val")) if '.DS' not in item]
        test_ids = [item for item in os.listdir(os.path.join(path, "test")) if '.DS' not in item]

        f = open(os.path.join(path, "train.json"), 'r')
        for line in f:
            tr_dict = json.loads(line)
        f = open(os.path.join(path, "val.json"), 'r')
        for line in f:
            val_dict = json.loads(line)
        f = open(os.path.join(path, "test.json"), 'r')
        for line in f:
            te_dict = json.loads(line)

        for img_id in train_ids:
            for caption_id in tr_dict[img_id]['captions']:
                train[0].append(tr_dict[img_id]['captions'][caption_id])
                train[1].append(1)
                train[2].append(int(img_id))
                # train[2].append(caption_id)
        for img_id in val_ids:
            for caption_id in val_dict[img_id]['captions']:
                val[0].append(val_dict[img_id]['captions'][caption_id])
                val[1].append(1)
                val[2].append(int(img_id))
                # val[2].append(caption_id)
        for img_id in test_ids:
            for caption_id in te_dict[img_id]['captions']:
                test[0].append(te_dict[img_id]['captions'][caption_id])
                test[1].append(1)
                test[2].append(int(img_id))
                # test[2].append(caption_id)

        log.info("Positive train samples: " + str(len(train[0])))



        ''' Shapeworld data '''

        
        f = open("/nfs/jsalt/home/roma/shapeworld/train.tsv", 'r')
        for line in f:
            items = line.strip().split('\t')
            train[0].append(items[0])
            train[1].append(int(items[1]))
            train[2].append(int(items[2]))

        f = open("/nfs/jsalt/home/roma/shapeworld/val.tsv", 'r')
        for line in f:
            items = line.strip().split('\t')
            val[0].append(items[0])
            val[1].append(int(items[1]))
            val[2].append(int(items[2]))

        f = open("/nfs/jsalt/home/roma/shapeworld/test.tsv", 'r')
        for line in f:
            items = line.strip().split('\t')
            test[0].append(items[0])
            test[1].append(int(items[1]))
            test[2].append(int(items[2]))

            
        r = 5
        train_ids = list(repeat(train_ids, r)); test_ids = list(repeat(test_ids, r)); val_ids = list(repeat(val_ids, r));
        train_ids = [item for sublist in train_ids for item in sublist]
        test_ids = [item for sublist in test_ids for item in sublist]
        val_ids = [item for sublist in val_ids for item in sublist]
        
        for img_id in train_ids:
            rand_id = img_id
            while (rand_id == img_id):
                rand_id = np.random.randint(len(train_ids), size=(1,1))[0][0]
            caption_id = np.random.randint(5, size=(1,1))[0][0]
            captions = tr_dict[train_ids[rand_id]]['captions']; caption_ids = list(captions.keys())
            caption = captions[caption_ids[caption_id]]
            train[0].append(caption); train[1].append(0); train[2].append(int(img_id))

        for img_id in val_ids:
            rand_id = img_id
            while (rand_id == img_id):
                rand_id = np.random.randint(len(val_ids), size=(1,1))[0][0]
            caption_id = np.random.randint(5, size=(1,1))[0][0]
            captions = val_dict[val_ids[rand_id]]['captions']; caption_ids = list(captions.keys())
            caption = captions[caption_ids[caption_id]]            
            val[0].append(caption); val[1].append(0); val[2].append(int(img_id))

        for img_id in test_ids:
            rand_id = img_id
            while (rand_id == img_id):
                rand_id = np.random.randint(len(test_ids), size=(1,1))[0][0]
            caption_id = np.random.randint(5, size=(1,1))[0][0]
            captions = te_dict[test_ids[rand_id]]['captions']; caption_ids = list(captions.keys())
            caption = captions[caption_ids[caption_id]]
            test[0].append(caption); test[1].append(0); test[2].append(int(img_id))
        

        


        #np.random.shuffle(train); np.random.shuffle(test); np.random.shuffle(val)
        
        log.info("All train samples: " + str(len(train[0])))

        self.tr_data = train
        self.val_data = val
        self.te_data = test
        self.train_data_text = train
        self.val_data_text = val
        self.test_data_text = test
        log.info('Train: ' + str(len(train)) + ' , Val: ' + str(len(val)) + ', Test: ' + str(len(test)))
        log.info("\tFinished loading MSCOCO data.")

class VAETask(SequenceGenerationTask):
    '''Variational Autoencoder (with corrupted input) Task'''

    def __init__(self, path, max_seq_len, name='MTTask'):
        super().__init__(name)
        self.scorer1 = Average()
        self.scorer2 = None
        self.val_metric = "%s_perplexity" % self.name
        self.val_metric_decreases = True
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0] + \
            self.train_data_text[2] + self.val_data_text[2]

    def load_data(self, path, max_seq_len):
        '''
        self.train_data_text = load_tsv(os.path.join(path, 'wmt_sample.txt'), max_seq_len,
                                        s1_idx=0, s2_idx=None, targ_idx=1,
                                        targ_fn=lambda t: t.split(' '))
        self.val_data_text = self.train_data_text; self.test_data_text = self.train_data_text
        '''
        self.train_data_text = load_tsv(os.path.join(path, 'train.txt'), max_seq_len,
                                        s1_idx=0, s2_idx=None, targ_idx=1,
                                        targ_fn=lambda t: t.split(' '))

        self.val_data_text = load_tsv(os.path.join(path, 'valid.txt'), max_seq_len,
                                      s1_idx=0, s2_idx=None, targ_idx=1,
                                      targ_fn=lambda t: t.split(' '))
        self.test_data_text = load_tsv(os.path.join(path, 'test.txt'), max_seq_len,
                                       s1_idx=0, s2_idx=None, targ_idx=1,
                                       targ_fn=lambda t: t.split(' '))
        log.info("\tFinished loading VAE data.")

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        ppl = self.scorer1.get_metric(reset)
        return {'perplexity': ppl}

class RecastNLITask(PairClassificationTask):
    ''' Task class for NLI Recast Data'''

    def __init__(self, path, max_seq_len, name="recast"):
        super(RecastNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len,
                        s1_idx=1, s2_idx=2, skip_rows=0, targ_idx=3)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len,
                        s1_idx=0, s2_idx=1, skip_rows=0, targ_idx=3)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                        s1_idx=1, s2_idx=2, skip_rows=0, targ_idx=3)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading recast probing data.")

class RecastPunTask(RecastNLITask):

    def __init__(self, path, max_seq_len, name="recast-puns"):
        super(RecastPunTask, self).__init__(path, max_seq_len, name)

class RecastNERTask(RecastNLITask):

    def __init__(self, path, max_seq_len, name="recast-ner"):
        super(RecastNERTask, self).__init__(path, max_seq_len, name)

class RecastVerbnetTask(RecastNLITask):

    def __init__(self, path, max_seq_len, name="recast-verbnet"):
        super(RecastVerbnetTask, self).__init__(path, max_seq_len, name)

class RecastVerbcornerTask(RecastNLITask):

    def __init__(self, path, max_seq_len, name="recast-verbcorner"):
        super(RecastVerbcornerTask, self).__init__(path, max_seq_len, name)

class RecastSentimentTask(RecastNLITask):

    def __init__(self, path, max_seq_len, name="recast-sentiment"):
        super(RecastSentimentTask, self).__init__(path, max_seq_len, name)

class RecastFactualityTask(RecastNLITask):

    def __init__(self, path, max_seq_len, name="recast-factuality"):
        super(RecastFactualityTask, self).__init__(path, max_seq_len, name)

class RecastWinogenderTask(RecastNLITask):

    def __init__(self, path, max_seq_len, name="recast-winogender"):
        super(RecastWinogenderTask, self).__init__(path, max_seq_len, name)

class RecastLexicosynTask(RecastNLITask):

    def __init__(self, path, max_seq_len, name="recast-lexicosyn"):
        super(RecastLexicosynTask, self).__init__(path, max_seq_len, name)

class RecastKGTask(RecastNLITask):

    def __init__(self, path, max_seq_len, name="recast-kg"):
        super(RecastKGTask, self).__init__(path, max_seq_len, name)



class TaggingTask(Task):
    ''' Generic tagging, one tag per word '''

    def __init__(self, name, num_tags):
        super().__init__(name)
        self.num_tags = num_tags + 2 # add unknown and padding
        ## TODO check if this is good metric
        self.scorer1 = Average()
        self.scorer2 = None
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = True
        self.target_indexer = {"words": SingleIdTokenIndexer(namespace="targets")} # TODO namespace

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
        ''' Process a tagging task '''
        inputs = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        targs = [TextField(list(map(Token, sent)), token_indexers=self.target_indexer) for sent in split[2]]
        # Might be better as LabelField? I don't know what these things mean
        instances = [Instance({"inputs": x, "targs": t}) for (x, t) in zip(inputs, targs)]
        return instances

class POSTaggingTask(TaggingTask):
    def __init__(self, path, max_seq_len, name="pos"):
        super().__init__(name, 45) # 45 tags
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.target_sentences = self.train_data_text[2] + self.val_data_text[2]


    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
        tr_data = load_tsv(os.path.join(path, "pos_45.train"), max_seq_len,
                           s1_idx=0, s2_idx=None, targ_idx=1, targ_fn=lambda t: t.split(' '))
        val_data = load_tsv(os.path.join(path, "pos_45.dev"), max_seq_len,
                            s1_idx=0, s2_idx=None, targ_idx=1, targ_fn=lambda t: t.split(' '))
        te_data = load_tsv(os.path.join(path, 'pos_45.test'), max_seq_len,
                           s1_idx=0, s2_idx=None, targ_idx=1, targ_fn=lambda t: t.split(' '))
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading POSTagging data.")



class CCGTaggingTask(TaggingTask):
    def __init__(self, path, max_seq_len, name="ccg"):
        super().__init__(name, 1363) # 1363 tags
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.target_sentences = self.train_data_text[2] + self.val_data_text[2]



    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
        tr_data = load_tsv(os.path.join(path, "ccg_1363.train"), max_seq_len,
                           s1_idx=0, s2_idx=None, targ_idx=1, targ_fn=lambda t: t.split(' '))
        val_data = load_tsv(os.path.join(path, "ccg_1363.dev"), max_seq_len,
                            s1_idx=0, s2_idx=None, targ_idx=1, targ_fn=lambda t: t.split(' '))
        te_data = load_tsv(os.path.join(path, 'ccg_1363.test'), max_seq_len,
                           s1_idx=0, s2_idx=None, targ_idx=1, targ_fn=lambda t: t.split(' '))
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading CCGTagging data.")









