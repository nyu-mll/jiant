"""Task definitions for NLI probing tasks."""
import logging as log
import os
from typing import Any, Dict, Iterable, List, Sequence, Type

from ..utils.data_loaders import load_tsv, process_sentence
from .registry import register_task
from .tasks import PairClassificationTask


@register_task("nps", rel_path="nps/")
class NPSTask(PairClassificationTask):
    def __init__(self, path, max_seq_len, name, **kw):
        super(NPSTask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        prob_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
            label_fn=targ_map.__getitem__,
            skip_rows=0,
        )

        self.train_data_text = self.val_data_text = self.test_data_text = prob_data
        log.info("\tFinished loading NP/S data.")


@register_task("nli-prob", rel_path="NLI-Prob/")
class NLITypeProbingTask(PairClassificationTask):
    """ Task class for Probing Task (NLI-type)"""

    def __init__(self, path, max_seq_len, name, probe_path="", **kw):
        super(NLITypeProbingTask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len, probe_path)
        self.sentences = self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, probe_path):
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        prob_data = load_tsv(
            data_file=os.path.join(path, probe_path),
            max_seq_len=max_seq_len,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
            label_fn=targ_map.__getitem__,
            skip_rows=0,
            tokenizer_name=self._tokenizer_name,
        )

        self.train_data_text = self.val_data_text = self.test_data_text = prob_data
        log.info("\tFinished loading NLI-type probing data.")


@register_task("nli-prob-negation", rel_path="NLI-Prob/")
class NLITypeProbingTaskNeg(PairClassificationTask):
    def __init__(self, path, max_seq_len, name, **kw):
        super(NLITypeProbingTaskNeg, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len):
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}

        prob_data = load_tsv(
            data_file=os.path.join(path),
            max_seq_len=max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=10,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
            tokenizer_name=self._tokenizer_name,
        )

        self.train_data_text = self.val_data_text = self.test_data_text = prob_data
        log.info("\tFinished loading negation data.")


@register_task("nli-prob-prepswap", rel_path="NLI-Prob/")
class NLITypeProbingTaskPrepswap(PairClassificationTask):
    def __init__(self, path, max_seq_len, name, **kw):
        super(NLITypeProbingTaskPrepswap, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len):
        prob_data = load_tsv(
            data_file=os.path.join(path, "all.prepswap.turk.newlabels.tsv"),
            max_seq_len=max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=0,
            skip_rows=0,
            tokenizer_name=self._tokenizer_name,
        )
        self.train_data_text = self.val_data_text = self.test_data_text = prob_data
        log.info("\tFinished loading preposition swap data.")


@register_task("nli-alt", rel_path="NLI-Prob/")
class NLITypeProbingAltTask(PairClassificationTask):
    """ Task class for Alt Probing Task (NLI-type), NLITypeProbingTask with different indices"""

    def __init__(self, path, max_seq_len, name, probe_path="", **kw):
        super(NLITypeProbingAltTask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len, probe_path)
        self.sentences = self.val_data_text[0] + self.val_data_text[1]

    def load_data(self, path, max_seq_len, probe_path):
        targ_map = {"0": 0, "1": 1, "2": 2}
        prob_data = load_tsv(
            data_file=os.path.join(path, probe_path),
            max_seq_len=max_seq_len,
            s1_idx=9,
            s2_idx=10,
            label_idx=1,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
            return_indices=True,
            tokenizer_name=self._tokenizer_name,
        )

        self.train_data_text = self.val_data_text = self.test_data_text = prob_data
        log.info("\tFinished loading NLI-alt probing data.")
