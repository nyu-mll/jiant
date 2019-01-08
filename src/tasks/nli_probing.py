"""Task definitions for NLI probing tasks."""
import logging as log
import os

from ..utils.utils import load_tsv, process_sentence, truncate

from typing import Iterable, Sequence, List, Dict, Any, Type

from .tasks import PairClassificationTask
from .registry import register_task

@register_task('nli-prob', rel_path='NLI-Prob/')
class NLITypeProbingTask(PairClassificationTask):
    ''' Task class for Probing Task (NLI-type)'''

    def __init__(self, path, max_seq_len, name, probe_path="probe_dummy.tsv",
                 **kw):
        super(NLITypeProbingTask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len, probe_path)
        #  self.use_classifier = 'mnli'  # use .conf params instead
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, probe_path):
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(os.path.join(path, 'train_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, targ_map=targ_map, skip_rows=0)
        val_data = load_tsv(os.path.join(path, probe_path), max_seq_len,
                            s1_idx=0, s2_idx=1, targ_idx=2, targ_map=targ_map, skip_rows=0)
        te_data = load_tsv(os.path.join(path, 'test_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, targ_map=targ_map, skip_rows=0)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading NLI-type probing data.")


@register_task('nli-prob-negation', rel_path='NLI-Prob/')
class NLITypeProbingTaskNeg(PairClassificationTask):

    def __init__(self, path, max_seq_len, name, probe_path="probe_dummy.tsv",
                 **kw):
        super(NLITypeProbingTaskNeg, self).__init__(name, n_classes=3,
                                                    **kw)
        self.load_data(path, max_seq_len, probe_path)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, probe_path):
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(os.path.join(path, 'train_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, skip_rows=0)
        val_data = load_tsv(os.path.join(path, 'lexnegs.tsv'), max_seq_len,
                            s1_idx=8, s2_idx=9, targ_idx=10, targ_map=targ_map, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, skip_rows=0)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading negation data.")


@register_task('nli-prob-prepswap', rel_path='NLI-Prob/')
class NLITypeProbingTaskPrepswap(PairClassificationTask):

    def __init__(self, path, max_seq_len, name, probe_path="probe_dummy.tsv",
                 **kw):
        super(NLITypeProbingTaskPrepswap, self).__init__(name, n_classes=3,
                                                         **kw)
        self.load_data(path, max_seq_len, probe_path)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, probe_path):
        tr_data = load_tsv(os.path.join(path, 'train_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, skip_rows=0)
        val_data = load_tsv(os.path.join(path, 'all.prepswap.turk.newlabels.tsv'), max_seq_len,
                            s1_idx=8, s2_idx=9, targ_idx=0, skip_rows=0)
        te_data = load_tsv(os.path.join(path, 'test_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, skip_rows=0)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading preposition swap data.")

@register_task('nli-alt', rel_path='NLI-Prob/')
class NLITypeProbingAltTask(NLITypeProbingTask):
    ''' Task class for Alt Probing Task (NLI-type), NLITypeProbingTask with different indices'''

    def __init__(self, path, max_seq_len, name, probe_path="probe_dummy.tsv",
                 **kw):
        super(NLITypeProbingTask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len, probe_path)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, probe_path):
        targ_map = {'0': 0, '1': 1, '2': 2}
        tr_data = load_tsv(os.path.join(path, 'train_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, targ_map=targ_map, skip_rows=0)
        val_data = load_tsv(
            os.path.join(
                path,
                probe_path),
            max_seq_len,
            idx_idx=0,
            s1_idx=9,
            s2_idx=10,
            targ_idx=1,
            targ_map=targ_map,
            skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test_dummy.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, targ_map=targ_map, skip_rows=0)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading NLI-alt probing data.")

