"""Task definitions for edge probing."""
import collections
import itertools
import logging as log
import os
from typing import Dict, Iterable, List, Sequence, Type

# Fields for instance processing
from allennlp.data import Instance
from allennlp.data.fields import ListField, MetadataField, SpanField
from allennlp.training.metrics import BooleanAccuracy, F1Measure

from ..allennlp_mods.correlation import FastMatthews
from ..allennlp_mods.multilabel_field import MultiLabelField
from ..utils import utils
from .registry import register_task  # global task registry
from .tasks import Task, sentence_to_text_field

##
# Class definitions for edge probing. See below for actual task registration.


class EdgeProbingTask(Task):
    """ Generic class for fine-grained edge probing.

    Acts as a classifier, but with multiple targets for each input text.

    Targets are of the form (span1, span2, label), where span1 and span2 are
    half-open token intervals [i, j).

    Subclass this for each dataset, or use register_task with appropriate kw
    args.
    """

    @property
    def _tokenizer_suffix(self):
        """ Suffix to make sure we use the correct source files,
        based on the given tokenizer.
        """
        if self.tokenizer_name:
            return ".retokenized." + self.tokenizer_name
        else:
            return ""

    def tokenizer_is_supported(self, tokenizer_name):
        """ Check if the tokenizer is supported for this task. """
        # Assume all tokenizers supported; if retokenized data not found
        # for this particular task, we'll just crash on file loading.
        return True

    def __init__(
        self,
        path: str,
        max_seq_len: int,
        name: str,
        label_file: str = None,
        files_by_split: Dict[str, str] = None,
        is_symmetric: bool = False,
        single_sided: bool = False,
        **kw,
    ):
        """Construct an edge probing task.

        path, max_seq_len, and name are passed by the code in preprocess.py;
        remaining arguments should be provided by a subclass constructor or via
        @register_task.

        Args:
            path: data directory
            max_seq_len: maximum sequence length (currently ignored)
            name: task name
            label_file: relative path to labels file
            files_by_split: split name ('train', 'val', 'test') mapped to
                relative filenames (e.g. 'train': 'train.json')
            is_symmetric: if true, span1 and span2 are assumed to be the same
                type and share parameters. Otherwise, we learn a separate
                projection layer and attention weight for each.
            single_sided: if true, only use span1.
        """
        super().__init__(name, **kw)

        assert label_file is not None
        assert files_by_split is not None
        self._files_by_split = {
            split: os.path.join(path, fname) + self._tokenizer_suffix
            for split, fname in files_by_split.items()
        }
        self.path = path
        self.label_file = label_file
        self.max_seq_len = max_seq_len
        self.is_symmetric = is_symmetric
        self.single_sided = single_sided

        self._iters_by_split = None
        self.all_labels = None
        self.n_classes = None

        # see add_task_label_namespace in preprocess.py
        self._label_namespace = self.name + "_labels"

        # Scorers
        #  self.acc_scorer = CategoricalAccuracy()  # multiclass accuracy
        self.mcc_scorer = FastMatthews()
        self.acc_scorer = BooleanAccuracy()  # binary accuracy
        self.f1_scorer = F1Measure(positive_label=1)  # binary F1 overall
        self.val_metric = "%s_f1" % self.name  # TODO: switch to MCC?
        self.val_metric_decreases = False

    def load_data(self):
        label_file = os.path.join(self.path, self.label_file)
        self.all_labels = list(utils.load_lines(label_file))
        self.n_classes = len(self.all_labels)

    @classmethod
    def _stream_records(cls, filename):
        skip_ctr = 0
        total_ctr = 0
        for record in utils.load_json_data(filename):
            total_ctr += 1
            # Skip records with empty targets.
            # TODO(ian): don't do this if generating negatives!
            if not record.get("targets", None):
                skip_ctr += 1
                continue
            yield record
        log.info(
            "Read=%d, Skip=%d, Total=%d from %s",
            total_ctr - skip_ctr,
            skip_ctr,
            total_ctr,
            filename,
        )

    @staticmethod
    def merge_preds(record: Dict, preds: Dict) -> Dict:
        """ Merge predictions into record, in-place.

        List-valued predictions should align to targets,
        and are attached to the corresponding target entry.

        Non-list predictions are attached to the top-level record.
        """
        record["preds"] = {}
        for target in record["targets"]:
            target["preds"] = {}
        for key, val in preds.items():
            if isinstance(val, list):
                assert len(val) == len(record["targets"])
                for i, target in enumerate(record["targets"]):
                    target["preds"][key] = val[i]
            else:
                # non-list predictions, attach to top-level preds
                record["preds"][key] = val
        return record

    def load_data(self):
        iters_by_split = collections.OrderedDict()
        for split, filename in self._files_by_split.items():
            #  # Lazy-load using RepeatableIterator.
            #  loader = functools.partial(utils.load_json_data,
            #                             filename=filename)
            #  iter = serialize.RepeatableIterator(loader)
            iter = list(self._stream_records(filename))
            iters_by_split[split] = iter
        self._iters_by_split = iters_by_split

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.

        Split should be one of 'train', 'val', or 'test'.
        """
        return self._iters_by_split[split]

    @classmethod
    def get_num_examples(cls, split_text):
        """ Return number of examples in the result of get_split_text.

        Subclass can override this if data is not stored in column format.
        """
        return len(split_text)

    @classmethod
    def _make_span_field(cls, s, text_field, offset=1):
        return SpanField(s[0] + offset, s[1] - 1 + offset, text_field)

    def _pad_tokens(self, tokens):
        """Pad tokens according to the current tokenization style."""
        if self.tokenizer_name.startswith("bert-"):
            # standard padding for BERT; see
            # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py#L85  # noqa
            return ["[CLS]"] + tokens + ["[SEP]"]
        else:
            return [utils.SOS_TOK] + tokens + [utils.EOS_TOK]

    def make_instance(self, record, idx, indexers) -> Type[Instance]:
        """Convert a single record to an AllenNLP Instance."""
        tokens = record["text"].split()  # already space-tokenized by Moses
        tokens = self._pad_tokens(tokens)
        text_field = sentence_to_text_field(tokens, indexers)

        d = {}
        d["idx"] = MetadataField(idx)

        d["input1"] = text_field

        d["span1s"] = ListField(
            [self._make_span_field(t["span1"], text_field, 1) for t in record["targets"]]
        )
        if not self.single_sided:
            d["span2s"] = ListField(
                [self._make_span_field(t["span2"], text_field, 1) for t in record["targets"]]
            )

        # Always use multilabel targets, so be sure each label is a list.
        labels = [utils.wrap_singleton_string(t["label"]) for t in record["targets"]]
        d["labels"] = ListField(
            [
                MultiLabelField(
                    label_set, label_namespace=self._label_namespace, skip_indexing=False
                )
                for label_set in labels
            ]
        )
        return Instance(d)

    def process_split(self, records, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _map_fn(r, idx):
            return self.make_instance(r, idx, indexers)

        return map(_map_fn, records, itertools.count())

    def get_all_labels(self) -> List[str]:
        return self.all_labels

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split, iter in self._iters_by_split.items():
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            for record in iter:
                yield record["text"].split()

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        metrics = {}
        metrics["mcc"] = self.mcc_scorer.get_metric(reset)
        metrics["acc"] = self.acc_scorer.get_metric(reset)
        precision, recall, f1 = self.f1_scorer.get_metric(reset)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        return metrics


##
# Task definitions. We call the register_task decorator explicitly so that we
# can group these in the code.
##


##
# Core probing tasks. as featured in the paper.
##
# Part-of-Speech tagging on OntoNotes.
register_task(
    "edges-pos-ontonotes",
    rel_path="edges/ontonotes-constituents",
    label_file="labels.pos.txt",
    files_by_split={
        "train": "consts_ontonotes_en_train.pos.json",
        "val": "consts_ontonotes_en_dev.pos.json",
        "test": "consts_ontonotes_en_test.pos.json",
    },
    single_sided=True,
)(EdgeProbingTask)
# Constituency labeling (nonterminals) on OntoNotes.
register_task(
    "edges-nonterminal-ontonotes",
    rel_path="edges/ontonotes-constituents",
    label_file="labels.nonterminal.txt",
    files_by_split={
        "train": "consts_ontonotes_en_train.nonterminal.json",
        "val": "consts_ontonotes_en_dev.nonterminal.json",
        "test": "consts_ontonotes_en_test.nonterminal.json",
    },
    single_sided=True,
)(EdgeProbingTask)
# Dependency edge labeling on English Web Treebank (UD).
register_task(
    "edges-dep-labeling-ewt",
    rel_path="edges/dep_ewt",
    label_file="labels.txt",
    files_by_split={
        "train": "train.edges.json",
        "val": "dev.edges.json",
        "test": "test.edges.json",
    },
    is_symmetric=False,
)(EdgeProbingTask)
# Entity type labeling on OntoNotes.
register_task(
    "edges-ner-ontonotes",
    rel_path="edges/ontonotes-ner",
    label_file="labels.txt",
    files_by_split={
        "train": "ner_ontonotes_en_train.json",
        "val": "ner_ontonotes_en_dev.json",
        "test": "ner_ontonotes_en_test.json",
    },
    single_sided=True,
)(EdgeProbingTask)
# SRL CoNLL 2012 (OntoNotes), formulated as an edge-labeling task.
register_task(
    "edges-srl-conll2012",
    rel_path="edges/srl_conll2012",
    label_file="labels.txt",
    files_by_split={
        "train": "train.edges.json",
        "val": "dev.edges.json",
        "test": "test.edges.json",
    },
    is_symmetric=False,
)(EdgeProbingTask)
# Re-processed version of edges-coref-ontonotes, via AllenNLP data loaders.
register_task(
    "edges-coref-ontonotes-conll",
    rel_path="edges/ontonotes-coref-conll",
    label_file="labels.txt",
    files_by_split={
        "train": "coref_conll_ontonotes_en_train.json",
        "val": "coref_conll_ontonotes_en_dev.json",
        "test": "coref_conll_ontonotes_en_test.json",
    },
    is_symmetric=False,
)(EdgeProbingTask)
# SPR1, as an edge-labeling task (multilabel).
register_task(
    "edges-spr1",
    rel_path="edges/spr1",
    label_file="labels.txt",
    files_by_split={"train": "spr1.train.json", "val": "spr1.dev.json", "test": "spr1.test.json"},
    is_symmetric=False,
)(EdgeProbingTask)
# SPR2, as an edge-labeling task (multilabel).
register_task(
    "edges-spr2",
    rel_path="edges/spr2",
    label_file="labels.txt",
    files_by_split={
        "train": "train.edges.json",
        "val": "dev.edges.json",
        "test": "test.edges.json",
    },
    is_symmetric=False,
)(EdgeProbingTask)
# Definite pronoun resolution. Two labels.
register_task(
    "edges-dpr",
    rel_path="edges/dpr",
    label_file="labels.txt",
    files_by_split={
        "train": "train.edges.json",
        "val": "dev.edges.json",
        "test": "test.edges.json",
    },
    is_symmetric=False,
)(EdgeProbingTask)
# Relation classification on SemEval 2010 Task8. 19 labels.
register_task(
    "edges-rel-semeval",
    rel_path="edges/semeval",
    label_file="labels.txt",
    files_by_split={"train": "train.0.85.json", "val": "dev.json", "test": "test.json"},
    is_symmetric=False,
)(EdgeProbingTask)

##
# New or experimental tasks.
##
# Relation classification on TACRED. 42 labels.
register_task(
    "edges-rel-tacred",
    rel_path="edges/tacred/rel",
    label_file="labels.txt",
    files_by_split={"train": "train.json", "val": "dev.json", "test": "test.json"},
    is_symmetric=False,
)(EdgeProbingTask)

##
# Older tasks or versions for backwards compatibility.
##
# Entity classification on TACRED. 17 labels.
# NOTE: these are probably silver labels from CoreNLP,
# so this is of limited use as a target.
register_task(
    "edges-ner-tacred",
    rel_path="edges/tacred/entity",
    label_file="labels.txt",
    files_by_split={"train": "train.json", "val": "dev.json", "test": "test.json"},
    single_sided=True,
)(EdgeProbingTask)
# SRL CoNLL 2005, formulated as an edge-labeling task.
register_task(
    "edges-srl-conll2005",
    rel_path="edges/srl_conll2005",
    label_file="labels.txt",
    files_by_split={
        "train": "train.edges.json",
        "val": "dev.edges.json",
        "test": "test.wsj.edges.json",
    },
    is_symmetric=False,
)(EdgeProbingTask)
# Coreference on OntoNotes corpus. Two labels.
register_task(
    "edges-coref-ontonotes",
    rel_path="edges/ontonotes-coref",
    label_file="labels.txt",
    files_by_split={
        "train": "train.edges.json",
        "val": "dev.edges.json",
        "test": "test.edges.json",
    },
    is_symmetric=False,
)(EdgeProbingTask)
# Entity type labeling on CoNLL 2003.
register_task(
    "edges-ner-conll2003",
    rel_path="edges/ner_conll2003",
    label_file="labels.txt",
    files_by_split={
        "train": "CoNLL-2003_train.json",
        "val": "CoNLL-2003_dev.json",
        "test": "CoNLL-2003_test.json",
    },
    single_sided=True,
)(EdgeProbingTask)
# Dependency edge labeling on UD treebank (GUM). Use 'ewt' version instead.
register_task(
    "edges-dep-labeling",
    rel_path="edges/dep",
    label_file="labels.txt",
    files_by_split={"train": "train.json", "val": "dev.json", "test": "test.json"},
    is_symmetric=False,
)(EdgeProbingTask)
# PTB constituency membership / labeling.
register_task(
    "edges-constituent-ptb",
    rel_path="edges/ptb-membership",
    label_file="labels.txt",
    files_by_split={"train": "ptb_train.json", "val": "ptb_dev.json", "test": "ptb_test.json"},
    single_sided=True,
)(EdgeProbingTask)
# Constituency membership / labeling on OntoNotes.
register_task(
    "edges-constituent-ontonotes",
    rel_path="edges/ontonotes-constituents",
    label_file="labels.txt",
    files_by_split={
        "train": "consts_ontonotes_en_train.json",
        "val": "consts_ontonotes_en_dev.json",
        "test": "consts_ontonotes_en_test.json",
    },
    single_sided=True,
)(EdgeProbingTask)
# CCG tagging (tokens only).
register_task(
    "edges-ccg-tag",
    rel_path="edges/ccg_tag",
    label_file="labels.txt",
    files_by_split={
        "train": "ccg.tag.train.json",
        "val": "ccg.tag.dev.json",
        "test": "ccg.tag.test.json",
    },
    single_sided=True,
)(EdgeProbingTask)
# CCG parsing (constituent labeling).
register_task(
    "edges-ccg-parse",
    rel_path="edges/ccg_parse",
    label_file="labels.txt",
    files_by_split={
        "train": "ccg.parse.train.json",
        "val": "ccg.parse.dev.json",
        "test": "ccg.parse.test.json",
    },
    single_sided=True,
)(EdgeProbingTask)
