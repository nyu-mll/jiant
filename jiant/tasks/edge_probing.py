"""Task definitions for edge probing."""
import collections
import itertools
import logging as log
import os
import torch
from typing import Dict, Iterable, List, Sequence, Type

# Fields for instance processing
from allennlp.data import Instance
from allennlp.data.fields import ListField, MetadataField, SpanField
from allennlp.training.metrics import BooleanAccuracy, F1Measure

from jiant.allennlp_mods.correlation import FastMatthews
from jiant.allennlp_mods.multilabel_field import MultiLabelField
from jiant.utils import utils
from jiant.tasks.registry import register_task  # global task registry
from jiant.tasks.tasks import Task, sentence_to_text_field

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
        self.label_file = os.path.join(self.path, label_file)
        self.max_seq_len = max_seq_len
        self.single_sided = single_sided

        # Placeholders; see self.load_data()
        self._iters_by_split = None
        self.all_labels = None
        self.n_classes = None

        # see add_task_label_namespace in preprocess.py
        self._label_namespace = self.name + "_labels"

        # Scorers
        self.mcc_scorer = FastMatthews()
        self.acc_scorer = BooleanAccuracy()  # binary accuracy
        self.f1_scorer = F1Measure(positive_label=1)  # binary F1 overall
        self.val_metric = "%s_f1" % self.name  # TODO: switch to MCC?
        self.val_metric_decreases = False

    def get_all_labels(self) -> List[str]:
        return self.all_labels

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
        self.all_labels = list(utils.load_lines(self.label_file))
        self.n_classes = len(self.all_labels)
        iters_by_split = collections.OrderedDict()
        for split, filename in self._files_by_split.items():
            #  # Lazy-load using RepeatableIterator.
            #  loader = functools.partial(utils.load_json_data,
            #                             filename=filename)
            #  iter = serialize.RepeatableIterator(loader)
            iter = list(self._stream_records(filename))
            iters_by_split[split] = iter
        self._iters_by_split = iters_by_split

    def update_metrics(self, out, batch):
        span_mask = batch["span1s"][:, :, 0] != -1
        logits = out["logits"][span_mask]
        labels = batch["labels"][span_mask]

        binary_preds = logits.ge(0).long()  # {0,1}

        # Matthews coefficient and accuracy computed on {0,1} labels.
        self.mcc_scorer(binary_preds, labels.long())
        self.acc_scorer(binary_preds, labels.long())

        # F1Measure() expects [total_num_targets, n_classes, 2]
        # to compute binarized F1.
        binary_scores = torch.stack([-1 * logits, logits], dim=2)
        self.f1_scorer(binary_scores, labels)

    def handle_preds(self, preds, batch):
        """Unpack preds into varying-length numpy arrays, return the non-masked preds in a list.

        Parameters
        ----------
            preds : [batch_size, num_targets, ...]
            batch : dict
                dict with key "span1s" having val w/ bool Tensor dim [batch_size, num_targets, ...].

        Returns
        -------
            non_masked_preds : list[np.ndarray]
                list of of pred np.ndarray selected by the corresponding row of span_mask.

        """
        masks = batch["span1s"][:, :, 0] != -1
        preds = preds.detach().cpu()
        masks = masks.detach().cpu()
        non_masked_preds = []
        for pred, mask in zip(torch.unbind(preds, dim=0), torch.unbind(masks, dim=0)):
            non_masked_preds.append(pred[mask].numpy())  # only non-masked predictions
        return non_masked_preds

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

    def make_instance(self, record, idx, indexers, model_preprocessing_interface) -> Type[Instance]:
        """Convert a single record to an AllenNLP Instance."""
        tokens = record["text"].split()  # already space-tokenized by Moses
        tokens = model_preprocessing_interface.boundary_token_fn(
            tokens
        )  # apply model-appropriate variants of [cls] and [sep].
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

    def process_split(
        self, records, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _map_fn(r, idx):
            return self.make_instance(r, idx, indexers, model_preprocessing_interface)

        return map(_map_fn, records, itertools.count())

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
    rel_path="edges/ontonotes/const/pos",
    label_file="labels.txt",
    files_by_split={"train": "train.json", "val": "development.json", "test": "test.json"},
    single_sided=True,
)(EdgeProbingTask)
# Constituency labeling (nonterminals) on OntoNotes.
register_task(
    "edges-nonterminal-ontonotes",
    rel_path="edges/ontonotes/const/nonterminal",
    label_file="labels.txt",
    files_by_split={"train": "train.json", "val": "development.json", "test": "test.json"},
    single_sided=True,
)(EdgeProbingTask)
# Dependency edge labeling on English Web Treebank (UD).
register_task(
    "edges-dep-ud-ewt",
    rel_path="edges/dep_ewt",
    label_file="labels.txt",
    files_by_split={
        "train": "en_ewt-ud-train.json",
        "val": "en_ewt-ud-dev.json",
        "test": "en_ewt-ud-test.json",
    },
)(EdgeProbingTask)
# Entity type labeling on OntoNotes.
register_task(
    "edges-ner-ontonotes",
    rel_path="edges/ontonotes/ner",
    label_file="labels.txt",
    files_by_split={"train": "train.json", "val": "development.json", "test": "test.json"},
    single_sided=True,
)(EdgeProbingTask)
# Semantic role labeling on OntoNotes.
register_task(
    "edges-srl-ontonotes",
    rel_path="edges/ontonotes/srl",
    label_file="labels.txt",
    files_by_split={"train": "train.json", "val": "development.json", "test": "test.json"},
)(EdgeProbingTask)
# Coreference on OntoNotes (single-sentence context).
register_task(
    "edges-coref-ontonotes",
    rel_path="edges/ontonotes/coref",
    label_file="labels.txt",
    files_by_split={"train": "train.json", "val": "development.json", "test": "test.json"},
)(EdgeProbingTask)
# SPR1, as an edge-labeling task (multilabel).
register_task(
    "edges-spr1",
    rel_path="edges/spr1",
    label_file="labels.txt",
    files_by_split={"train": "spr1.train.json", "val": "spr1.dev.json", "test": "spr1.test.json"},
)(EdgeProbingTask)
# SPR2, as an edge-labeling task (multilabel).
register_task(
    "edges-spr2",
    rel_path="edges/spr2",
    label_file="labels.txt",
    files_by_split={
        "train": "edges.train.json",
        "val": "edges.dev.json",
        "test": "edges.test.json",
    },
)(EdgeProbingTask)
# Definite pronoun resolution. Two labels.
register_task(
    "edges-dpr",
    rel_path="edges/dpr",
    label_file="labels.txt",
    files_by_split={"train": "train.json", "val": "dev.json", "test": "test.json"},
)(EdgeProbingTask)
# Relation classification on SemEval 2010 Task8. 19 labels.
register_task(
    "edges-rel-semeval",
    rel_path="edges/semeval",
    label_file="labels.txt",
    files_by_split={"train": "train.0.85.json", "val": "dev.json", "test": "test.json"},
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
)(EdgeProbingTask)
