'''Define the tasks and code for loading their data.

- As much as possible, following the existing task hierarchy structure.
- When inheriting, be sure to write and call load_data.
- Set all text data as an attribute, task.sentences (List[List[str]])
- Each task's val_metric should be name_metric, where metric is returned by get_metrics()
'''
import os
import logging as log
import ipdb as pdb

from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Average

from utils import load_tsv


class Task():
    '''Generic class for a task

    Methods and attributes:
        - load_data: load dataset from a path and create splits
        - yield dataset for training
        - dataset size
        - validate and test

    Outside the task:
        - process: pad and indexify data given a mapping
        - optimizer
    '''

    def __init__(self, name):
        self.name = name

    def load_data(self, path, max_seq_len):
        ''' Load data from path and create splits. '''
        raise NotImplementedError

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        raise NotImplementedError


class SingleClassificationTask(Task):
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


class PairClassificationTask(Task):
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


class PairRegressionTask(Task):
    ''' Generic sentence pair classification '''

    def __init__(self, name):
        super().__init__(name)
        self.scorer1 = Average()  # for average MSE
        self.scorer2 = None
        self.val_metric = "%s_mse" % self.name
        self.val_metric_decreases = True

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        mse = self.scorer1.get_metric(reset)
        return {'mse': mse}


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
        raise NotImplementedError

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        raise NotImplementedError


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
        ppl = self.scorer1.get_metric(reset)
        return {'perplexity': ppl}


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
                toks = row.strip().split()[:max_seq_len]
                if not toks:
                    continue
                data.append(['<SOS>'] + toks + ['<EOS>'])
        return data


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


class CoLATask(SingleClassificationTask):
    '''Class for Warstdadt acceptability task'''

    def __init__(self, path, max_seq_len, name="acceptability"):
        ''' '''
        super(CoLATask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.val_metric = "%s_mcc" % self.name
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

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''
        super(
            MultiNLIFictionTask,
            self).__init__(
            path,
            max_seq_len,
            genre="fiction",
            name="mnli-fiction")


class MultiNLISlateTask(MultiNLISingleGenreTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''
        super(MultiNLISlateTask, self).__init__(path, max_seq_len, genre="slate", name="mnli-slate")


class MultiNLIGovernmentTask(MultiNLISingleGenreTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''
        super(
            MultiNLIGovernmentTask,
            self).__init__(
            path,
            max_seq_len,
            genre="government",
            name="mnli-government")


class MultiNLITelephoneTask(MultiNLISingleGenreTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''
        super(
            MultiNLITelephoneTask,
            self).__init__(
            path,
            max_seq_len,
            genre="telephone",
            name="mnli-telephone")


class MultiNLITravelTask(MultiNLISingleGenreTask):
    ''' Task class for Multi-Genre Natural Language Inference, Fiction genre.'''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''
        super(
            MultiNLITravelTask,
            self).__init__(
            path,
            max_seq_len,
            genre="travel",
            name="mnli-travel")


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
        self.scorer1 = Average()
        self.scorer2 = Average()
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
        log.info("\tFinished loading RTE{1,2,3}.")


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
