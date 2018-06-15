'''Define the tasks and code for loading them'''
import os
import logging as log
from abc import ABCMeta, abstractmethod
import ipdb as pdb

from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Average

from utils import load_tsv

class Task():
    '''Abstract class for a task

    Methods and attributes:
        - load_data: load dataset from a path and create splits
        - yield dataset for training
        - dataset size
        - validate and test

    Outside the task:
        - process: pad and indexify data given a mapping
        - optimizer
    '''
    __metaclass__ = ABCMeta

    def __init__(self, name, n_classes):
        self.name = name
        self.n_classes = n_classes
        self.train_data_text, self.val_data_text, self.test_data_text = \
            None, None, None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.pair_input = 1
        self.categorical = 1
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer1 = CategoricalAccuracy()
        self.scorer2 = None

    @abstractmethod
    def load_data(self, path, max_seq_len):
        '''
        Load data from path and create splits.
        '''
        raise NotImplementedError

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        return {'accuracy': acc}

class PairClassificationTask(Task):
    ''' Generic sentence pair classification '''
    def __init_(self, name, n_classes):
        super().__init__(name, n_classes)

class SingleClassificationTask(Task):
    ''' Generic sentence pair classification '''
    def __init_(self, name, n_classes):
        super().__init__(name, n_classes)

class PairRegressionTask(Task):
    ''' Generic sentence pair classification '''
    def __init_(self, name, n_classes):
        super().__init__(name, n_classes)

class SequenceGenerationTask(Task):
    ''' Generic sentence generation task '''
    def __init_(self, name, n_classes):
        super().__init__(name, n_classes)

class RankingTask(Task):
    ''' Generic sentence ranking task, given some input '''
    def __init_(self, name, n_classes):
        super().__init__(name, n_classes)

class QQPTask(PairClassificationTask):
    ''' Task class for Quora Question Pairs. '''

    def __init__(self, path, max_seq_len, name="qqp"):
        super().__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.scorer2 = F1Measure(1)
        self.val_metric = "%s_acc_f1" # TODO(Alex): handle non acc validation

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
        prc, rcl, f1 = self.scorer2.get_metric(reset)
        return {'accuracy': (acc + f1) / 2, 'acc': acc, 'f1': f1,
                'precision': prc, 'recall': rcl}

class SNLITask(PairClassificationTask):
    ''' Task class for Stanford Natural Language Inference '''

    def __init__(self, path, max_seq_len, name="snli"):
        ''' Args: '''
        super(SNLITask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)
        self.scorer2 = None

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

    def get_metrics(self, reset=False):
        ''' No F1 '''
        return {'accuracy': self.scorer1.get_metric(reset)}

class MultiNLITask(PairClassificationTask):
    ''' Task class for Multi-Genre Natural Language Inference '''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''
        super(MultiNLITask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)
        self.scorer2 = None

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
        te_data = [m + mm + d for m, mm, d in \
                    zip(te_matched_data, te_mismatched_data, te_diagnostic_data)]
        te_data[3] = list(range(len(te_data[3])))

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading MNLI data.")

    def get_metrics(self, reset=False):
        ''' No F1 '''
        return {'accuracy': self.scorer1.get_metric(reset)}

class MRPCTask(PairClassificationTask):
    ''' Task class for Microsoft Research Paraphase Task.  '''

    def __init__(self, path, max_seq_len, name="mrpc"):
        ''' '''
        super(MRPCTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.scorer2 = F1Measure(1)

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
        log.info("\tFinished loading MSRP data.")

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        prc, rcl, f1 = self.scorer2.get_metric(reset)
        return {'accuracy': (acc + f1) / 2, 'acc': acc, 'f1': f1,
                'precision': prc, 'recall': rcl}

class STSBTask(PairRegressionTask):
    ''' Task class for Sentence Textual Similarity Benchmark.  '''
    def __init__(self, path, max_seq_len, name="sts_benchmark"):
        ''' '''
        super(STSBTask, self).__init__(name, 1)
        self.categorical = 0
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Average()
        self.scorer2 = Average()
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''
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
        # NB: I think I call it accuracy b/c something weird in training
        return {'accuracy': self.scorer1.get_metric(reset),
                'spearmanr': self.scorer2.get_metric(reset)}

class SSTTask(SingleClassificationTask):
    ''' Task class for Stanford Sentiment Treebank.  '''
    def __init__(self, path, max_seq_len, name="sst"):
        ''' '''
        super(SSTTask, self).__init__(name, 2)
        self.pair_input = 0
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''
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

class RTETask(PairClassificationTask):
    ''' Task class for Recognizing Textual Entailment 1, 2, 3, 5 '''

    def __init__(self, path, max_seq_len, name="rte"):
        ''' '''
        super(RTETask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

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
        log.info("\tFinished loading RTE{1,2,3}.")

class QNLITask(PairClassificationTask):
    '''Task class for SQuAD NLI'''
    def __init__(self, path, max_seq_len, name="squad"):
        super(QNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

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

class CoLATask(SingleClassificationTask):
    '''Class for Warstdadt acceptability task'''
    def __init__(self, path, max_seq_len, name="acceptability"):
        ''' '''
        super(CoLATask, self).__init__(name, 2)
        self.pair_input = 0
        self.load_data(path, max_seq_len)
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Average()
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
        # NB: I think I call it accuracy b/c something weird in training
        return {'accuracy': self.scorer1.get_metric(reset),
                'acc': self.scorer2.get_metric(reset)}

class WNLITask(PairClassificationTask):
    '''Class for Winograd NLI task'''
    def __init__(self, path, max_seq_len, name="winograd"):
        ''' '''
        super(WNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

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
