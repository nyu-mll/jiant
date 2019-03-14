"""Task definitions for NLI probing tasks."""
import logging as log
import os
from typing import Iterable, Sequence, List, Dict, Any, Type

from .tasks import PairClassificationTask
from .registry import register_task
from ..utils.data_loaders import load_tsv, process_sentence

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
        label_fn = targ_map.__getitem__
        tr_data = load_tsv(data_file=os.path.join(path, 'train_dummy.tsv'), max_seq_len=max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=None, label_fn=label_fn, skip_rows=0,
                           tokenizer_name=self._tokenizer_name)
        val_data = load_tsv(data_file=os.path.join(path, probe_path), max_seq_len=max_seq_len,
                            s1_idx=0, s2_idx=1, label_idx=2, label_fn=label_fn, skip_rows=0,
                            tokenizer_name=self._tokenizer_name)
        te_data = load_tsv(data_file=os.path.join(path, 'test_dummy.tsv'), max_seq_len=max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=None, label_fn=label_fn, skip_rows=0,
                           tokenizer_name=self._tokenizer_name)

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
        label_fn = targ_map.__getitem__
        tr_data = load_tsv(data_file=os.path.join(path, 'train_dummy.tsv'), max_seq_len=max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=None, skip_rows=0,
                           tokenizer_name=self._tokenizer_name)
        # TODO(?): should this use probe_path?
        val_data = load_tsv(data_file=os.path.join(path, 'lexnegs.tsv'), max_seq_len=max_seq_len,
                            s1_idx=8, s2_idx=9, label_idx=10, label_fn=label_fn, skip_rows=1,
                            tokenizer_name=self._tokenizer_name)
        te_data = load_tsv(data_file=os.path.join(path, 'test_dummy.tsv'), max_seq_len=max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=None, skip_rows=0,
                           tokenizer_name=self._tokenizer_name)

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
        tr_data = load_tsv(data_file=os.path.join(path, 'train_dummy.tsv'), max_seq_len=max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=None, skip_rows=0,
                           tokenizer_name=self._tokenizer_name)
        # TODO(?): should this use probe_path?
        val_data = load_tsv(data_file=os.path.join(path, 'all.prepswap.turk.newlabels.tsv'),
                            max_seq_len=max_seq_len, s1_idx=8, s2_idx=9, label_idx=0, skip_rows=0,
                            tokenizer_name=self._tokenizer_name)
        te_data = load_tsv(data_file=os.path.join(path, 'test_dummy.tsv'), max_seq_len=max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=None, skip_rows=0,
                           tokenizer_name=self._tokenizer_name)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading preposition swap data.")

@register_task('nli-alt', rel_path='NLI-Prob/')
class NLITypeProbingAltTask(NLITypeProbingTask):
    ''' Task class for Alt Probing Task (NLI-type), NLITypeProbingTask with different indices'''

    def __init__(self, path, max_seq_len, name, probe_path="probe_dummy.tsv",
                 **kw):
        super(NLITypeProbingAltTask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len, probe_path)
        self.sentences = self.train_data_text[0] + self.train_data_text[1] + \
            self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, probe_path):
        targ_map = {'0': 0, '1': 1, '2': 2}
        label_fn = targ_map.__getitem__
        tr_data = load_tsv(data_file=os.path.join(path, 'train_dummy.tsv'), max_seq_len=max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=None, label_fn=label_fn, skip_rows=0,
                           tokenizer_name=self._tokenizer_name)
        val_data = load_tsv(data_file=os.path.join(path, probe_path), max_seq_len=max_seq_len,
                            s1_idx=9, s2_idx=10, label_idx=1, label_fn=label_fn, skip_rows=1,
                            return_indices=True, tokenizer_name=self._tokenizer_name)
        te_data = load_tsv(data_file=os.path.join(path, 'test_dummy.tsv'), max_seq_len=max_seq_len,
                           s1_idx=1, s2_idx=2, label_idx=None, label_fn=label_fn, skip_rows=0,
                           tokenizer_name=self._tokenizer_name)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading NLI-alt probing data.")
