"""Task definitions for NLI probing tasks."""
import logging as log
import os

from jiant.utils.data_loaders import load_tsv
from jiant.tasks.registry import register_task
from jiant.tasks.tasks import PairClassificationTask


@register_task("nps", rel_path="nps/")
class NPSTask(PairClassificationTask):
    def __init__(self, path, max_seq_len, name, **kw):
        super(NPSTask, self).__init__(name, n_classes=3, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        prob_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
            label_fn=targ_map.__getitem__,
            skip_rows=0,
        )
        self.train_data_text = self.val_data_text = self.test_data_text = prob_data
        self.sentences = self.val_data_text[0] + self.val_data_text[1]
        log.info("\tFinished loading NP/S data.")


@register_task("nli-prob", rel_path="NLI-Prob/")
class NLITypeProbingTask(PairClassificationTask):
    """ Task class for Probing Task (NLI-type)"""

    def __init__(self, path, max_seq_len, name, probe_path="", **kw):
        super(NLITypeProbingTask, self).__init__(name, n_classes=3, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.probe_path = probe_path

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        prob_data = load_tsv(
            data_file=os.path.join(self.path, self.probe_path),
            max_seq_len=self.max_seq_len,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
            label_fn=targ_map.__getitem__,
            skip_rows=0,
            tokenizer_name=self._tokenizer_name,
        )

        self.train_data_text = self.val_data_text = self.test_data_text = prob_data
        self.sentences = self.val_data_text[0] + self.val_data_text[1]
        log.info("\tFinished loading NLI-type probing data.")


@register_task("nli-alt", rel_path="NLI-Prob/")
class NLITypeProbingAltTask(NLITypeProbingTask):
    """ Task class for Alt Probing Task (NLI-type), NLITypeProbingTask with different indices"""

    def __init__(self, path, max_seq_len, name, probe_path="", **kw):
        super(NLITypeProbingAltTask, self).__init__(
            name=name, path=path, max_seq_len=max_seq_len, **kw
        )
        self.path = path
        self.max_seq_len = max_seq_len
        self.probe_path = probe_path

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        targ_map = {"0": 0, "1": 1, "2": 2}
        prob_data = load_tsv(
            data_file=os.path.join(self.path, self.probe_path),
            max_seq_len=self.max_seq_len,
            s1_idx=9,
            s2_idx=10,
            label_idx=1,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
            return_indices=True,
            tokenizer_name=self._tokenizer_name,
        )

        self.train_data_text = self.val_data_text = self.test_data_text = prob_data
        self.sentences = self.val_data_text[0] + self.val_data_text[1]
        log.info("\tFinished loading NLI-alt probing data.")


@register_task(
    "nli-prob-prep",
    rel_path="FunctionWordsProbing/nli-prep/",
    s1_idx=8,
    s2_idx=9,
    label_idx=0,
    skip_rows=1,
)
@register_task(
    "nli-prob-negation",
    rel_path="FunctionWordsProbing/nli-negation/",
    s1_idx=8,
    s2_idx=9,
    label_idx=0,
    skip_rows=1,
)
@register_task(
    "nli-prob-spatial",
    rel_path="FunctionWordsProbing/nli-spatial/",
    idx_idx=0,
    s1_idx=9,
    s2_idx=10,
    label_idx=0,
    skip_rows=1,
)
@register_task(
    "nli-prob-quant",
    rel_path="FunctionWordsProbing/nli-quant/",
    idx_idx=0,
    s1_idx=9,
    s2_idx=10,
    label_idx=0,
    skip_rows=1,
)
@register_task(
    "nli-prob-comp",
    rel_path="FunctionWordsProbing/nli-comp/",
    idx_idx=0,
    s1_idx=9,
    s2_idx=10,
    label_idx=0,
    skip_rows=1,
)
class NLIProbingTask(PairClassificationTask):
    """ Task class for NLI-type Probing Task
    This probing task only have evaluation set, need to share classifier with NLI-type task.
    At present, 5 tasks are registered under this class,
    nli-prob-prep tests model's understanding of Prepositions
    nli-prob-negation tests that of Negation
    nli-prob-spatial tests that of Spatial Expressions
    nli-prob-quant tests that of Quantification
    nli-prob-comp tests that of Comparatives
    """

    def __init__(
        self,
        path,
        max_seq_len,
        name,
        targ_map={"0": 0, "1": 1, "2": 2},
        idx_idx=None,
        s1_idx=None,
        s2_idx=None,
        label_idx=None,
        skip_rows=0,
        **kw,
    ):
        super(NLIProbingTask, self).__init__(name, n_classes=3, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.eval_only_task = True

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.targ_map = targ_map
        self.s1_idx = s1_idx
        self.s2_idx = s2_idx
        self.label_idx = label_idx
        self.skip_rows = skip_rows

    def load_data(self):
        targ_map = self.targ_map
        prob_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
            label_fn=targ_map.__getitem__,
            skip_rows=0,
        )
        self.train_data_text = self.val_data_text = self.test_data_text = prob_data
        self.sentences = self.val_data_text[0] + self.val_data_text[1]
        log.info("\tFinished loading NLI-type probing data on %s." % self.name)
