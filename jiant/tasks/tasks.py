import collections
import itertools
import json
import logging as log
import os
from typing import Any, Dict, Iterable, List, Sequence, Type
import random

import numpy as np
import pandas as pd
import torch


# Fields for instance processing
from allennlp.data import Instance, Token, vocabulary
from allennlp.data.fields import (
    LabelField,
    ListField,
    MetadataField,
    MultiLabelField,
    SpanField,
    TextField,
)
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy, F1Measure
from sklearn.metrics import mean_squared_error

from jiant.allennlp_mods.correlation import Correlation
from jiant.allennlp_mods.numeric_field import NumericField
from jiant.utils import utils
from jiant.utils.data_loaders import (
    get_tag_list,
    load_diagnostic_tsv,
    load_span_data,
    load_tsv,
    tokenize_and_truncate,
    load_pair_nli_jsonl,
)
from jiant.utils.tokenizers import get_tokenizer
from jiant.utils.retokenize import get_aligner_fn
from jiant.tasks.registry import register_task  # global task registry
from jiant.metrics.winogender_metrics import GenderParity
from jiant.metrics.nli_metrics import NLITwoClassAccuracy

"""Define the tasks and code for loading their data.

- As much as possible, following the existing task hierarchy structure.
- When inheriting, be sure to write and call load_data.
- Set all text data as an attribute, task.sentences (List[List[str]])
- Each task's val_metric should be name_metric, where metric is returned by
get_metrics(): e.g. if task.val_metric = task_name + "_accuracy", then
task.get_metrics() should return {"accuracy": accuracy_val, ... }
"""


UNK_TOK_ALLENNLP = "@@UNKNOWN@@"
UNK_TOK_ATOMIC = "UNKNOWN"  # an unk token that won't get split by tokenizers


def sentence_to_text_field(sent: Sequence[str], indexers: Any):
    """ Helper function to map a sequence of tokens into a sequence of
    AllenNLP Tokens, then wrap in a TextField with the given indexers """
    return TextField(list(map(Token, sent)), token_indexers=indexers)


def atomic_tokenize(
    sent: str, atomic_tok: str, nonatomic_toks: List[str], max_seq_len: int, tokenizer_name: str
):
    """ Replace tokens that will be split by tokenizer with a
    placeholder token. Tokenize, and then substitute the placeholder
    with the *first* nonatomic token in the list. """
    for nonatomic_tok in nonatomic_toks:
        sent = sent.replace(nonatomic_tok, atomic_tok)
    sent = tokenize_and_truncate(tokenizer_name, sent, max_seq_len)
    sent = [nonatomic_toks[0] if t == atomic_tok else t for t in sent]
    return sent


def process_single_pair_task_split(
    split,
    indexers,
    model_preprocessing_interface,
    is_pair=True,
    classification=True,
    label_namespace="labels",
    is_symmetrical_pair=False,
    skip_indexing=True,
):
    """
    Convert a dataset of sentences into padded sequences of indices. Shared
    across several classes.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - indexers ()
        - model_preprocessing_interface: packed information from model that effects the task data
        - is_pair (Bool)
        - classification (Bool)
        - is_symmetrical_pair (Bool) whether reverse the sentences in a pair will change the result,
            if this is true, and the model allows uses_mirrored_pair, a mirrored pair will be added
            to the input

    Returns:
        - instances (Iterable[Instance]): an iterable of AllenNLP Instances with fields
    """
    # check here if using bert to avoid passing model info to tasks

    def _make_instance(input1, input2, labels, idx):
        d = {}
        d["sent1_str"] = MetadataField(" ".join(input1))
        if model_preprocessing_interface.model_flags["uses_pair_embedding"] and is_pair:
            inp = model_preprocessing_interface.boundary_token_fn(input1, input2)
            d["inputs"] = sentence_to_text_field(inp, indexers)
            d["sent2_str"] = MetadataField(" ".join(input2))
            if (
                model_preprocessing_interface.model_flags["uses_mirrored_pair"]
                and is_symmetrical_pair
            ):
                inp_m = model_preprocessing_interface.boundary_token_fn(input2, input1)
                d["inputs_m"] = sentence_to_text_field(inp_m, indexers)
        else:
            d["input1"] = sentence_to_text_field(
                model_preprocessing_interface.boundary_token_fn(input1), indexers
            )
            if input2:
                d["input2"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(input2), indexers
                )
                d["sent2_str"] = MetadataField(" ".join(input2))
        if classification:
            d["labels"] = LabelField(
                labels, label_namespace=label_namespace, skip_indexing=skip_indexing
            )
        else:
            d["labels"] = NumericField(labels)

        d["idx"] = LabelField(idx, label_namespace="idxs_tags", skip_indexing=True)

        return Instance(d)

    split = list(split)
    if not is_pair:  # dummy iterator for input2
        split[1] = itertools.repeat(None)
    if len(split) < 4:  # counting iterator for idx
        assert len(split) == 3
        split.append(itertools.count())

    # Map over columns: input1, (input2), labels, idx
    instances = map(_make_instance, *split)
    return instances  # lazy iterator


def create_subset_scorers(count, scorer_type, **args_to_scorer):
    """
    Create a list scorers of designated type for each "coarse__fine" tag.
    This function is only used by tasks that need evalute results on tags,
    and should be called after loading all the splits.

    Parameters:
        count: N_tag, number of different "coarse__fine" tags
        scorer_type: which scorer to use
        **args_to_scorer: arguments passed to the scorer
    Returns:
        scorer_list: a list of N_tag scorer object
    """
    scorer_list = [scorer_type(**args_to_scorer) for _ in range(count)]
    return scorer_list


def update_subset_scorers(scorer_list, estimations, labels, tagmask):
    """
    Add the output and label of one minibatch to the subset scorer objects.
    This function is only used by tasks that need evalute results on tags,
    and should be called every minibatch when task.scorer are updated.

    Parameters:
        scorer_list: a list of N_tag scorer object
        estimations: a (bs, *) tensor, model estimation
        labels: a (bs, *) tensor, ground truth
        tagmask: a (bs, N_tag) 0-1 tensor, indicating tags of each sample
    """
    for tid, scorer in enumerate(scorer_list):
        subset_idx = torch.nonzero(tagmask[:, tid]).squeeze(dim=1)
        subset_estimations = estimations[subset_idx]
        subset_labels = labels[subset_idx]
        if len(subset_idx) > 0:
            scorer(subset_estimations, subset_labels)
    return


def collect_subset_scores(scorer_list, metric_name, tag_list, reset=False):
    """
    Get the scorer measures of each tag.
    This function is only used by tasks that need evalute results on tags,
    and should be called in get_metrics.

    Parameters:
        scorer_list: a list of N_tag scorer object
        metric_name: string, name prefix for this group
        tag_list: "coarse__fine" tag strings
    Returns:
        subset_scores: a dictionary from subset tags to scores
        reset:
    """
    subset_scores = {
        "%s_%s" % (metric_name, tag_str): scorer.get_metric(reset)
        for tag_str, scorer in zip(tag_list, scorer_list)
    }
    return subset_scores


class Task(object):
    """Generic class for a task

    Methods and attributes:
        - load_data: load dataset from a path and create splits
        - get_metrics:

    Outside the task:
        - process: pad and indexify data given a mapping
        - optimizer
    """

    def __init__(self, name, tokenizer_name):
        self.name = name
        self._tokenizer_name = tokenizer_name
        self.scorers = []
        self.eval_only_task = False
        self.sentences = None
        self.example_counts = None
        self.contributes_to_aggregate_score = True

    def load_data(self):
        """ Load data from path and create splits. """
        raise NotImplementedError

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        yield from self.sentences

    def count_examples(self, splits=["train", "val", "test"]):
        """ Count examples in the dataset. """
        self.example_counts = {}
        for split in splits:
            st = self.get_split_text(split)
            count = self.get_num_examples(st)
            self.example_counts[split] = count

    @property
    def tokenizer_name(self):
        return self._tokenizer_name

    @property
    def n_train_examples(self):
        return self.example_counts["train"]

    @property
    def n_val_examples(self):
        return self.example_counts["val"]

    def get_split_text(self, split: str):
        """ Get split text, typically as list of columns.

        Split should be one of 'train', 'val', or 'test'.
        """
        return getattr(self, "%s_data_text" % split)

    def get_num_examples(self, split_text):
        """ Return number of examples in the result of get_split_text.

        Subclass can override this if data is not stored in column format.
        """
        return len(split_text[0])

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict:
        """ Get metrics specific to the task. """
        raise NotImplementedError

    def get_scorers(self):
        return self.scorers

    def update_metrics(self, out, batch):
        raise NotImplementedError

    def handle_preds(self, preds, batch):
        """
        Function that does task-specific processing of predictions.
        """
        return preds


class ClassificationTask(Task):
    """ General classification task """

    pass


class RegressionTask(Task):
    """ General regression task """

    pass


class SingleClassificationTask(ClassificationTask):
    """ Generic sentence pair classification """

    def __init__(self, name, n_classes, **kw):
        super().__init__(name, **kw)
        self.n_classes = n_classes
        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(
            split, indexers, model_preprocessing_interface, is_pair=False
        )

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = out["labels"]
        assert len(self.get_scorers()) > 0, "Please specify a score metric"
        for scorer in self.get_scorers():
            scorer(logits, labels)


class PairClassificationTask(ClassificationTask):
    """ Generic sentence pair classification """

    def __init__(self, name, n_classes, **kw):
        super().__init__(name, **kw)
        assert n_classes > 0
        self.n_classes = n_classes
        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(
            split, indexers, model_preprocessing_interface, is_pair=True
        )

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = out["labels"]
        assert len(self.get_scorers()) > 0, "Please specify a score metric"
        for scorer in self.get_scorers():
            scorer(logits, labels)


class PairRegressionTask(RegressionTask):
    """ Generic sentence pair classification """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.n_classes = 1
        self.scorer1 = Average()  # for average MSE
        self.scorers = [self.scorer1]
        self.val_metric = "%s_mse" % self.name
        self.val_metric_decreases = True

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        mse = self.scorer1.get_metric(reset)
        return {"mse": mse}

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(
            split, indexers, model_preprocessing_interface, is_pair=True, classification=False
        )

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = out["labels"]
        assert len(self.get_scorers()) > 0, "Please specify a score metric"
        for scorer in self.get_scorers():
            scorer(logits, labels)


class PairOrdinalRegressionTask(RegressionTask):
    """ Generic sentence pair ordinal regression.
        Currently just doing regression but added new class
        in case we find a good way to implement ordinal regression with NN"""

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.n_classes = 1
        self.scorer1 = Average()  # for average MSE
        self.scorer2 = Correlation("spearman")
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_1-mse" % self.name
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        mse = self.scorer1.get_metric(reset)
        spearmanr = self.scorer2.get_metric(reset)
        return {"1-mse": 1 - mse, "mse": mse, "spearmanr": spearmanr}

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(
            split, indexers, model_preprocessing_interface, is_pair=True, classification=False
        )

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = out["labels"]
        assert len(self.get_scorers()) > 0, "Please specify a score metric"
        for scorer in self.get_scorers():
            scorer(logits, labels)


class SequenceGenerationTask(Task):
    """ Generic sentence generation task """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.scorer1 = Average()  # for average BLEU or something
        self.scorers = [self.scorer1]
        self.val_metric = "%s_bleu" % self.name
        self.val_metric_decreases = False
        log.warning(
            "BLEU scoring is turned off (current code in progress)."
            "Please use outputed prediction files to score offline"
        )

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        bleu = self.scorer1.get_metric(reset)
        return {"bleu": bleu}

    def update_metrics(self, out, batch):
        # currently don't support metrics for regression task
        # TODO(Yada): support them!
        return


class RankingTask(Task):
    """ Generic sentence ranking task, given some input """

    pass


@register_task("sst", rel_path="SST-2/")
class SSTTask(SingleClassificationTask):
    """ Task class for Stanford Sentiment Treebank.  """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SSTTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        """ Load data """
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=0,
            s2_idx=None,
            label_idx=1,
            skip_rows=1,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=0,
            s2_idx=None,
            label_idx=1,
            skip_rows=1,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=None,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        log.info("\tFinished loading SST data.")


@register_task("npi-adv-li", rel_path="NPI/probing/adverbs/licensor")
@register_task("npi-adv-sc", rel_path="NPI/probing/adverbs/scope_with_licensor")
@register_task("npi-adv-pr", rel_path="NPI/probing/adverbs/npi_present")
@register_task("npi-cond-li", rel_path="NPI/probing/conditionals/licensor")
@register_task("npi-cond-sc", rel_path="NPI/probing/conditionals/scope_with_licensor")
@register_task("npi-cond-pr", rel_path="NPI/probing/conditionals/npi_present")
@register_task("npi-negdet-li", rel_path="NPI/probing/determiner_negation_biclausal/licensor")
@register_task(
    "npi-negdet-sc", rel_path="NPI/probing/determiner_negation_biclausal/scope_with_licensor"
)
@register_task("npi-negdet-pr", rel_path="NPI/probing/determiner_negation_biclausal/npi_present")
@register_task("npi-negsent-li", rel_path="NPI/probing/sentential_negation_biclausal/licensor")
@register_task(
    "npi-negsent-sc", rel_path="NPI/probing/sentential_negation_biclausal/scope_with_licensor"
)
@register_task("npi-negsent-pr", rel_path="NPI/probing/sentential_negation_biclausal/npi_present")
@register_task("npi-only-li", rel_path="NPI/probing/only/licensor")
@register_task("npi-only-sc", rel_path="NPI/probing/only/scope_with_licensor")
@register_task("npi-only-pr", rel_path="NPI/probing/only/npi_present")
@register_task("npi-qnt-li", rel_path="NPI/probing/quantifiers/licensor")
@register_task("npi-qnt-sc", rel_path="NPI/probing/quantifiers/scope_with_licensor")
@register_task("npi-qnt-pr", rel_path="NPI/probing/quantifiers/npi_present")
@register_task("npi-ques-li", rel_path="NPI/probing/questions/licensor")
@register_task("npi-ques-sc", rel_path="NPI/probing/questions/scope_with_licensor")
@register_task("npi-ques-pr", rel_path="NPI/probing/questions/npi_present")
@register_task("npi-quessmp-li", rel_path="NPI/probing/simplequestions/licensor")
@register_task("npi-quessmp-sc", rel_path="NPI/probing/simplequestions/scope_with_licensor")
@register_task("npi-quessmp-pr", rel_path="NPI/probing/simplequestions/npi_present")
@register_task("npi-sup-li", rel_path="NPI/probing/superlative/licensor")
@register_task("npi-sup-sc", rel_path="NPI/probing/superlative/scope_with_licensor")
@register_task("npi-sup-pr", rel_path="NPI/probing/superlative/npi_present")
@register_task("cola-npi-adv", rel_path="NPI/splits/adverbs")
@register_task("cola-npi-cond", rel_path="NPI/splits/conditionals")
@register_task("cola-npi-negdet", rel_path="NPI/splits/determiner_negation_biclausal")
@register_task("cola-npi-negsent", rel_path="NPI/splits/sentential_negation_biclausal")
@register_task("cola-npi-only", rel_path="NPI/splits/only")
@register_task("cola-npi-ques", rel_path="NPI/splits/questions")
@register_task("cola-npi-quessmp", rel_path="NPI/splits/simplequestions")
@register_task("cola-npi-qnt", rel_path="NPI/splits/quantifiers")
@register_task("cola-npi-sup", rel_path="NPI/splits/superlative")
@register_task("all-cola-npi", rel_path="NPI/combs/all_env")
@register_task("wilcox-npi", rel_path="NPI/wilcox")
@register_task("hd-cola-npi-adv", rel_path="NPI/combs/minus_adverbs")
@register_task("hd-cola-npi-cond", rel_path="NPI/combs/minus_conditionals")
@register_task("hd-cola-npi-negdet", rel_path="NPI/combs/minus_determiner_negation_biclausal")
@register_task("hd-cola-npi-negsent", rel_path="NPI/combs/minus_sentential_negation_biclausal")
@register_task("hd-cola-npi-only", rel_path="NPI/combs/minus_only")
@register_task("hd-cola-npi-ques", rel_path="NPI/combs/minus_questions")
@register_task("hd-cola-npi-quessmp", rel_path="NPI/combs/minus_simplequestions")
@register_task("hd-cola-npi-qnt", rel_path="NPI/combs/minus_quantifiers")
@register_task("hd-cola-npi-sup", rel_path="NPI/combs/minus_superlative")
class CoLANPITask(SingleClassificationTask):
    """Class for NPI-related task; same with Warstdadt acceptability task but outputs labels for
       test-set
       Note: Used for an NYU seminar, data not yet public"""

    def __init__(self, path, max_seq_len, name, **kw):
        super(CoLANPITask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        # self.scorer1 = Average()
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]

    def load_data(self):
        """Load the data"""
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test_full.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        log.info("\tFinished loading NPI Data.")

    def get_metrics(self, reset=False):
        return {"mcc": self.scorer1.get_metric(reset), "accuracy": self.scorer2.get_metric(reset)}

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = out["labels"]
        logits, labels = logits.detach(), labels.detach()
        _, preds = logits.max(dim=1)
        self.scorer1(preds, labels)
        self.scorer2(logits, labels)
        return


@register_task("cola", rel_path="CoLA/")
class CoLATask(SingleClassificationTask):
    """Class for Warstdadt acceptability task"""

    def __init__(self, path, max_seq_len, name, **kw):
        super(CoLATask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        # self.scorer1 = Average()
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]

    def load_data(self):
        """Load the data"""
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=None,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        log.info("\tFinished loading CoLA.")

    def get_metrics(self, reset=False):
        return {"mcc": self.scorer1.get_metric(reset), "accuracy": self.scorer2.get_metric(reset)}

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = out["labels"]
        logits, labels = logits.detach(), labels.detach()
        _, preds = logits.max(dim=1)
        self.scorer1(preds, labels)
        self.scorer2(logits, labels)
        return


@register_task("cola-analysis", rel_path="CoLA/")
class CoLAAnalysisTask(SingleClassificationTask):
    def __init__(self, path, max_seq_len, name, **kw):
        super(CoLAAnalysisTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.tag_list = None
        self.tag_scorers1 = None
        self.tag_scorers2 = None

        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]

    def load_data(self):
        """Load the data"""
        # Load data from tsv
        tag_vocab = vocabulary.Vocabulary(counter=None)
        tr_data = load_tsv(
            tokenizer_name=self._tokenizer_name,
            data_file=os.path.join(self.path, "train_analysis.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            tag2idx_dict={"Domain": 1},
            tag_vocab=tag_vocab,
        )
        val_data = load_tsv(
            tokenizer_name=self._tokenizer_name,
            data_file=os.path.join(self.path, "dev_analysis.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            tag2idx_dict={
                "Domain": 1,
                "Simple": 4,
                "Pred": 5,
                "Adjunct": 6,
                "Arg Types": 7,
                "Arg Altern": 8,
                "Imperative": 9,
                "Binding": 10,
                "Question": 11,
                "Comp Clause": 12,
                "Auxillary": 13,
                "to-VP": 14,
                "N, Adj": 15,
                "S-Syntax": 16,
                "Determiner": 17,
                "Violations": 18,
            },
            tag_vocab=tag_vocab,
        )
        te_data = load_tsv(
            tokenizer_name=self._tokenizer_name,
            data_file=os.path.join(self.path, "test_analysis.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            tag2idx_dict={"Domain": 1},
            tag_vocab=tag_vocab,
        )
        self.train_data_text = tr_data[:1] + tr_data[2:]
        self.val_data_text = val_data[:1] + val_data[2:]
        self.test_data_text = te_data[:1] + te_data[2:]
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        # Create score for each tag from tag-index dict
        self.tag_list = get_tag_list(tag_vocab)
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=Correlation, corr_type="matthews"
        )
        self.tag_scorers2 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy
        )
        log.info("\tFinished loading CoLA sperate domain.")

    def process_split(self, split, indexers, model_preprocessing_interface):
        def _make_instance(input1, labels, tagids):
            """ from multiple types in one column create multiple fields """
            d = {}
            d["input1"] = sentence_to_text_field(
                model_preprocessing_interface.boundary_token_fn(input1), indexers
            )
            d["sent1_str"] = MetadataField(" ".join(input1))
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            d["tagmask"] = MultiLabelField(
                tagids, label_namespace="tags", skip_indexing=True, num_labels=len(self.tag_list)
            )
            return Instance(d)

        instances = map(_make_instance, *split)
        return instances  # lazy iterator

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = out["labels"]
        tagmask = batch.get("tagmask", None)
        logits, labels = logits.detach(), labels.detach()
        _, preds = logits.max(dim=1)
        self.scorer1(preds, labels)
        self.scorer2(logits, labels)
        if tagmask is not None:
            update_subset_scorers(self.tag_scorers1, preds, labels, tagmask)
            update_subset_scorers(self.tag_scorers2, logits, labels, tagmask)
        return

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""

        collected_metrics = {
            "mcc": self.scorer1.get_metric(reset),
            "accuracy": self.scorer2.get_metric(reset),
        }
        collected_metrics.update(
            collect_subset_scores(self.tag_scorers1, "mcc", self.tag_list, reset)
        )
        collected_metrics.update(
            collect_subset_scores(self.tag_scorers2, "accuracy", self.tag_list, reset)
        )
        return collected_metrics


@register_task("qqp", rel_path="QQP/")
@register_task("qqp-alt", rel_path="QQP/")  # second copy for different params
class QQPTask(PairClassificationTask):
    """ Task class for Quora Question Pairs. """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer2 = F1Measure(1)
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_acc_f1" % name
        self.val_metric_decreases = False

    def load_data(self):
        """Process the dataset located at data_file."""

        def label_fn(x):
            if x == "":
                return 0
            else:
                return int(x)

        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=4,
            label_idx=5,
            label_fn=label_fn,
            skip_rows=1,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=4,
            label_idx=5,
            label_fn=label_fn,
            skip_rows=1,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading QQP data.")

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        pcs, rcl, f1 = self.scorer2.get_metric(reset)
        return {
            "acc_f1": (acc + f1) / 2,
            "accuracy": acc,
            "f1": f1,
            "precision": pcs,
            "recall": rcl,
        }

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(
            split, indexers, model_preprocessing_interface, is_pair=True, is_symmetrical_pair=True
        )


@register_task("mrpc", rel_path="MRPC/")
class MRPCTask(PairClassificationTask):
    """Task class for Microsoft Research Paraphase Task."""

    def __init__(self, path, max_seq_len, name, **kw):
        super(MRPCTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer2 = F1Measure(1)
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_acc_f1" % name
        self.val_metric_decreases = False

    def load_data(self):
        """ Process the dataset located at path.  """
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=4,
            label_idx=0,
            skip_rows=1,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=4,
            label_idx=0,
            skip_rows=1,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=4,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading MRPC data.")

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        pcs, rcl, f1 = self.scorer2.get_metric(reset)
        return {
            "acc_f1": (acc + f1) / 2,
            "accuracy": acc,
            "f1": f1,
            "precision": pcs,
            "recall": rcl,
        }

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(
            split, indexers, model_preprocessing_interface, is_pair=True, is_symmetrical_pair=True
        )


@register_task("sts-b", rel_path="STS-B/")
# second copy for different params
@register_task("sts-b-alt", rel_path="STS-B/")
class STSBTask(PairRegressionTask):
    """ Task class for Sentence Textual Similarity Benchmark.  """

    def __init__(self, path, max_seq_len, name, **kw):
        super(STSBTask, self).__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = Correlation("pearson")
        self.scorer2 = Correlation("spearman")
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_corr" % self.name
        self.val_metric_decreases = False

    def load_data(self):
        """ Load data """
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            skip_rows=1,
            s1_idx=7,
            s2_idx=8,
            label_idx=9,
            label_fn=lambda x: float(x) / 5,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            skip_rows=1,
            s1_idx=7,
            s2_idx=8,
            label_idx=9,
            label_fn=lambda x: float(x) / 5,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=7,
            s2_idx=8,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading STS Benchmark data.")

    def get_metrics(self, reset=False):
        pearsonr = self.scorer1.get_metric(reset)
        spearmanr = self.scorer2.get_metric(reset)
        return {"corr": (pearsonr + spearmanr) / 2, "pearsonr": pearsonr, "spearmanr": spearmanr}

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(
            split,
            indexers,
            model_preprocessing_interface,
            is_pair=True,
            classification=False,
            is_symmetrical_pair=True,
        )


@register_task("snli", rel_path="SNLI/")
class SNLITask(PairClassificationTask):
    """ Task class for Stanford Natural Language Inference """

    def __init__(self, path, max_seq_len, name, **kw):
        """ Do stuff """
        super(SNLITask, self).__init__(name, n_classes=3, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        """ Process the dataset located at path.  """
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}

        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=7,
            s2_idx=8,
            label_idx=10,
            skip_rows=1,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=7,
            s2_idx=8,
            label_idx=10,
            skip_rows=1,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=7,
            s2_idx=8,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading SNLI data.")


@register_task("adversarial-nli-a1", rel_path="AdversarialNLI/", datasets=["R1"])
@register_task("adversarial-nli-a2", rel_path="AdversarialNLI/", datasets=["R2"])
@register_task("adversarial-nli-a3", rel_path="AdversarialNLI/", datasets=["R3"])
@register_task("adversarial-nli", rel_path="AdversarialNLI/", datasets=["R1", "R2", "R3"])
class AdversarialNLITask(PairClassificationTask):
    """Task class for use with Adversarial Natural Language Inference dataset.

    Configures a 3-class PairClassificationTask using Adversarial NLI data.
    Requires original ANLI dataset file structure under the relative path.
    Data: https://dl.fbaipublicfiles.com/anli/anli_v0.1.zip
    Paper: https://arxiv.org/abs/1910.14599

    Attributes:
        path (str): AdversarialNLI path relative to JIANT_DATA_DIR
        max_seq_len (int): max tokens allowed in a sequence
        train_data_text (list[list[str], list[str], list[int]]):
            list of lists of context, hypothesis, and target training data
        val_data_text (list[list[str], list[str], list[int]]):
            list of lists of context, hypothesis, and target val data
        test_data_text (list[list[str], list[str], list[int]]):
            list of lists of context, hypothesis, and target test data
        datasets (list[str]): list of sub-datasets used in task (e.g., R1)
        sentences (list): list of all (tokenized) context and hypothesis
            texts from train and val data.
    """

    def __init__(self, path, max_seq_len, name, datasets, **kw):
        """Initialize an AdversarialNLITask task.

        Args:
            path (str): AdversarialNLI path relative to the data dir
            max_seq_len (int): max tokens allowed in a sequence
            name (str): task name, specified in @register_task
            datasets (list[str]): list of ANLI sub-datasets used in task
        """
        super(AdversarialNLITask, self).__init__(name, n_classes=3, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None
        self.datasets = datasets

    def _read_data(self, path: str) -> pd.core.frame.DataFrame:
        """Read json, tokenize text, encode labels as int, return dataframe."""
        df = pd.read_json(path_or_buf=path, encoding="UTF-8", lines=True)
        # for ANLI datasets n=neutral, e=entailment, c=contradiction
        df["target"] = df["label"].map({"n": 0, "e": 1, "c": 2})
        tokenizer = get_tokenizer(self._tokenizer_name)
        df["context"] = df["context"].apply(tokenizer.tokenize)
        df["hypothesis"] = df["hypothesis"].apply(tokenizer.tokenize)
        return df[["context", "hypothesis", "target"]]

    def load_data(self):
        """Read, preprocess and load data into an AdversarialNLITask.

        Assumes original dataset file structure under `self.rel_path`.
        Loads only the datasets (e.g., "R1") specified in the `datasets` attr.
        Populates task train_, val_, test_data_text and `sentence` attr.
        """
        train_dfs, val_dfs, test_dfs = [], [], []
        for dataset in self.datasets:
            train_dfs.append(self._read_data(os.path.join(self.path, dataset, "train.jsonl")))
            val_dfs.append(self._read_data(os.path.join(self.path, dataset, "dev.jsonl")))
            test_dfs.append(self._read_data(os.path.join(self.path, dataset, "test.jsonl")))
        train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
        val_df = pd.concat(val_dfs, axis=0, ignore_index=True)
        test_df = pd.concat(test_dfs, axis=0, ignore_index=True)

        self.train_data_text = [
            train_df["context"].tolist(),
            train_df["hypothesis"].tolist(),
            train_df["target"].tolist(),
        ]
        self.val_data_text = [
            val_df["context"].tolist(),
            val_df["hypothesis"].tolist(),
            val_df["target"].tolist(),
        ]
        self.test_data_text = [
            test_df["context"].tolist(),
            test_df["hypothesis"].tolist(),
            test_df["target"].tolist(),
        ]

        self.sentences = (
            train_df["context"].tolist()
            + train_df["hypothesis"].tolist()
            + val_df["context"].tolist()
            + val_df["hypothesis"].tolist()
        )

        log.info("\tFinished loading ANLI data: " + self.name)


@register_task("mnli", rel_path="MNLI/")
# Alternate version with a modified evaluation metric. For use in transfer evaluations on
# two-class test sets like RTE. Example config override:
#   pretrain_tasks = mnli, target_tasks = \"mnli-two,rte\", rte += {use_classifier = mnli-two}
@register_task("mnli-two", rel_path="MNLI/", two_class_evaluation=True)
# Second copy that can be assigned separate task-specific config options.
@register_task("mnli-alt", rel_path="MNLI/")
@register_task("mnli-fiction", rel_path="MNLI/", genre="fiction")
@register_task("mnli-slate", rel_path="MNLI/", genre="slate")
@register_task("mnli-government", rel_path="MNLI/", genre="government")
@register_task("mnli-telephone", rel_path="MNLI/", genre="telephone")
@register_task("mnli-travel", rel_path="MNLI/", genre="travel")
class MultiNLITask(PairClassificationTask):
    """ Task class for Multi-Genre Natural Language Inference. """

    def __init__(self, path, max_seq_len, name, genre=None, two_class_evaluation=False, **kw):
        """Set up the MNLI task object.

        When genre is set to one of the ten MNLI genres, only examples matching that genre will be
        loaded in any split. That may result in some of the sections (train, dev mismatched, ...)
        being empty.

        When two_class_evaluation is set, merge the contradiction and neutral labels, for both
        predictions and gold labels, in the metric when evaluating on this task.
        """
        super(MultiNLITask, self).__init__(name, n_classes=3, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.genre = genre
        if two_class_evaluation:
            self.scorer1 = NLITwoClassAccuracy()
            self.scorers = [self.scorer1]

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        """Process the dataset located at path."""
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=11,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )

        # Warning to anyone who edits this: The reference label is column *15*,
        # not 11 as above.
        val_matched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev_matched.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=15,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )
        val_mismatched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev_mismatched.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=15,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )
        val_data = [m + mm for m, mm in zip(val_matched_data, val_mismatched_data)]
        val_data = tuple(val_data)

        te_matched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test_matched.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=8,
            s2_idx=9,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )
        te_mismatched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test_mismatched.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=8,
            s2_idx=9,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )
        te_data = [m + mm for m, mm in zip(te_matched_data, te_mismatched_data)]

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading MNLI data.")


@register_task("mnli-ho", rel_path="MNLI/")
@register_task("mnli-two-ho", rel_path="MNLI/", two_class_evaluation=True)
@register_task("mnli-fiction-ho", rel_path="MNLI/", genre="fiction")
@register_task("mnli-slate-ho", rel_path="MNLI/", genre="slate")
@register_task("mnli-government-ho", rel_path="MNLI/", genre="government")
@register_task("mnli-telephone-ho", rel_path="MNLI/", genre="telephone")
@register_task("mnli-travel-ho", rel_path="MNLI/", genre="travel")
class MultiNLIHypothesisOnlyTask(SingleClassificationTask):
    """ Task class for MultiNLI hypothesis-only classification. """

    def __init__(self, path, max_seq_len, name, genre=None, two_class_evaluation=False, **kw):
        """Set up the MNLI-HO task object.

        When genre is set to one of the ten MNLI genres, only examples matching that genre will be
        loaded in any split. That may result in some of the sections (train, dev mismatched, ...)
        being empty.

        When two_class_evaluation is set, merge the contradiction and neutral labels, for both
        predictions and gold labels, in the metric when evaluating on this task.
        """
        super(MultiNLIHypothesisOnlyTask, self).__init__(name, n_classes=3, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.genre = genre

        if two_class_evaluation:
            self.scorer1 = NLITwoClassAccuracy()
            self.scorers = [self.scorer1]

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        """Process the dataset located at path."""
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=9,
            s2_idx=None,
            label_idx=11,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )

        # Warning to anyone who edits this: The reference label is column *15*,
        # not 11 as above.
        val_matched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev_matched.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=9,
            s2_idx=None,
            label_idx=15,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )
        val_mismatched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev_mismatched.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=9,
            s2_idx=None,
            label_idx=15,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )
        val_data = [m + mm for m, mm in zip(val_matched_data, val_mismatched_data)]
        val_data = tuple(val_data)

        te_matched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test_matched.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=9,
            s2_idx=None,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )
        te_mismatched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test_mismatched.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=9,
            s2_idx=None,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=self.genre,
        )
        te_data = [m + mm for m, mm in zip(te_matched_data, te_mismatched_data)]

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        log.info("\tFinished loading MNLI-HO data.")


# GLUE diagnostic (3-class NLI), expects TSV
@register_task("glue-diagnostic", rel_path="MNLI/", n_classes=3)
# GLUE diagnostic (2 class NLI, merging neutral and contradiction into not_entailment)
@register_task("glue-diagnostic-binary", rel_path="RTE/", n_classes=2)
class GLUEDiagnosticTask(PairClassificationTask):
    """ Task class for GLUE diagnostic data """

    def __init__(self, path, max_seq_len, name, n_classes, **kw):
        super().__init__(name, n_classes, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.ix_to_lex_sem_dic = None
        self.ix_to_pr_ar_str_dic = None
        self.ix_to_logic_dic = None
        self.ix_to_knowledge_dic = None
        self._scorer_all_mcc = None
        self._scorer_all_acc = None

        self.contributes_to_aggregate_score = False
        self.eval_only_task = True

    def load_data(self):
        """load diagnostics data. The tags for every column are loaded as indices.
        They will be converted to bools in preprocess_split function"""

        # Will create separate scorer for every tag. tag_group is the name of the
        # column it will have its own scorer
        def create_score_function(scorer, arg_to_scorer, tags_dict, tag_group):
            setattr(self, "scorer__%s" % tag_group, scorer(arg_to_scorer))
            for index, tag in tags_dict.items():
                # 0 is missing value
                if index == 0:
                    continue
                setattr(self, "scorer__%s__%s" % (tag_group, tag), scorer(arg_to_scorer))

        if self.n_classes == 2:
            targ_map = {"neutral": 0, "entailment": 1, "contradiction": 0}
        elif self.n_classes == 3:
            targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        else:
            raise ValueError("Invalid number of classes for NLI task")

        diag_data_dic = load_diagnostic_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "diagnostic-full.tsv"),
            max_seq_len=self.max_seq_len,
            s1_col="Premise",
            s2_col="Hypothesis",
            label_col="Label",
            label_fn=targ_map.__getitem__,
            skip_rows=1,
        )

        self.ix_to_lex_sem_dic = diag_data_dic["ix_to_lex_sem_dic"]
        self.ix_to_pr_ar_str_dic = diag_data_dic["ix_to_pr_ar_str_dic"]
        self.ix_to_logic_dic = diag_data_dic["ix_to_logic_dic"]
        self.ix_to_knowledge_dic = diag_data_dic["ix_to_knowledge_dic"]

        # Train, val, test splits are same. We only need one split but the code
        # probably expects all splits to be present.
        self.train_data_text = (
            diag_data_dic["sents1"],
            diag_data_dic["sents2"],
            diag_data_dic["targs"],
            diag_data_dic["idxs"],
            diag_data_dic["lex_sem"],
            diag_data_dic["pr_ar_str"],
            diag_data_dic["logic"],
            diag_data_dic["knowledge"],
        )
        self.val_data_text = self.train_data_text
        self.test_data_text = self.train_data_text
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading diagnostic data.")

        # TODO: use FastMatthews instead to save memory.
        create_score_function(Correlation, "matthews", self.ix_to_lex_sem_dic, "lex_sem")
        create_score_function(Correlation, "matthews", self.ix_to_pr_ar_str_dic, "pr_ar_str")
        create_score_function(Correlation, "matthews", self.ix_to_logic_dic, "logic")
        create_score_function(Correlation, "matthews", self.ix_to_knowledge_dic, "knowledge")
        self._scorer_all_mcc = Correlation("matthews")  # score all examples according to MCC
        self._scorer_all_acc = CategoricalAccuracy()  # score all examples according to acc
        log.info("\tFinished creating score functions for diagnostic data.")

    def update_metrics(self, out, batch):
        self.update_diagnostic_metrics(out["logits"], batch["labels"], batch)

    def update_diagnostic_metrics(self, logits, labels, batch):
        # Updates scorer for every tag in a given column (tag_group) and also the
        # the scorer for the column itself.
        _, preds = logits.max(dim=1)

        def update_scores_for_tag_group(ix_to_tags_dic, tag_group):
            for ix, tag in ix_to_tags_dic.items():
                # 0 is for missing tag so here we use it to update scorer for the column
                # itself (tag_group).
                if ix == 0:
                    # This will contain 1s on positions where at least one of the tags of this
                    # column is present.
                    mask = batch[tag_group]
                    scorer_str = "scorer__%s" % tag_group
                # This branch will update scorers of individual tags in the
                # column
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
                sub_preds = preds[indices_to_pull[:, 0]]
                scorer = getattr(self, scorer_str)
                scorer(sub_preds, sub_labels)
            return

        # Updates scorers for each tag.
        update_scores_for_tag_group(self.ix_to_lex_sem_dic, "lex_sem")
        update_scores_for_tag_group(self.ix_to_pr_ar_str_dic, "pr_ar_str")
        update_scores_for_tag_group(self.ix_to_logic_dic, "logic")
        update_scores_for_tag_group(self.ix_to_knowledge_dic, "knowledge")
        self._scorer_all_mcc(preds, labels)
        self._scorer_all_acc(logits, labels)

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def create_labels_from_tags(fields_dict, ix_to_tag_dict, tag_arr, tag_group):
            # If there is something in this row then tag_group should be set to
            # 1.
            is_tag_group = 1 if len(tag_arr) != 0 else 0
            fields_dict[tag_group] = LabelField(
                is_tag_group, label_namespace=tag_group + "_tags", skip_indexing=True
            )
            # For every possible tag in the column set 1 if the tag is present for
            # this example, 0 otherwise.
            for ix, tag in ix_to_tag_dict.items():
                if ix == 0:
                    continue
                is_present = 1 if ix in tag_arr else 0
                fields_dict["%s__%s" % (tag_group, tag)] = LabelField(
                    is_present, label_namespace="%s__%s_tags" % (tag_group, tag), skip_indexing=True
                )
            return

        def _make_instance(input1, input2, label, idx, lex_sem, pr_ar_str, logic, knowledge):
            """ from multiple types in one column create multiple fields """
            d = {}
            if model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                inp = model_preprocessing_interface.boundary_token_fn(input1, input2)
                d["inputs"] = sentence_to_text_field(inp, indexers)
            else:
                d["input1"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(input1), indexers
                )
                d["input2"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(input2), indexers
                )
            d["labels"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idx_tags", skip_indexing=True)
            d["sent1_str"] = MetadataField(" ".join(input1))
            d["sent2_str"] = MetadataField(" ".join(input2))

            # adds keys to dict "d" for every possible type in the column
            create_labels_from_tags(d, self.ix_to_lex_sem_dic, lex_sem, "lex_sem")
            create_labels_from_tags(d, self.ix_to_pr_ar_str_dic, pr_ar_str, "pr_ar_str")
            create_labels_from_tags(d, self.ix_to_logic_dic, logic, "logic")
            create_labels_from_tags(d, self.ix_to_knowledge_dic, knowledge, "knowledge")

            return Instance(d)

        instances = map(_make_instance, *split)
        #  return list(instances)
        return instances  # lazy iterator

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        collected_metrics = {}

        def collect_metrics(ix_to_tag_dict, tag_group):
            for index, tag in ix_to_tag_dict.items():
                # Index 0 is used for missing data, here it will be used for score of the
                # whole category.
                if index == 0:
                    scorer_str = "scorer__%s" % tag_group
                    scorer = getattr(self, scorer_str)
                    collected_metrics["%s" % (tag_group)] = scorer.get_metric(reset)
                else:
                    scorer_str = "scorer__%s__%s" % (tag_group, tag)
                    scorer = getattr(self, scorer_str)
                    collected_metrics["%s__%s" % (tag_group, tag)] = scorer.get_metric(reset)

        collect_metrics(self.ix_to_lex_sem_dic, "lex_sem")
        collect_metrics(self.ix_to_pr_ar_str_dic, "pr_ar_str")
        collect_metrics(self.ix_to_logic_dic, "logic")
        collect_metrics(self.ix_to_knowledge_dic, "knowledge")
        collected_metrics["all_mcc"] = self._scorer_all_mcc.get_metric(reset)
        collected_metrics["accuracy"] = self._scorer_all_acc.get_metric(reset)
        return collected_metrics


# SuperGLUE diagnostic (2-class NLI), expects JSONL
@register_task("broadcoverage-diagnostic", rel_path="RTE/diagnostics")
class BroadCoverageDiagnosticTask(GLUEDiagnosticTask):
    """ Class for SuperGLUE broad coverage (linguistics, commonsense, world knowledge)
        diagnostic task """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(path, max_seq_len, name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.n_classes = 2

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.ix_to_lex_sem_dic = None
        self.ix_to_pr_ar_str_dic = None
        self.ix_to_logic_dic = None
        self.ix_to_knowledge_dic = None
        self._scorer_all_mcc = None
        self._scorer_all_acc = None
        self.eval_only_task = True

    def load_data(self):
        """load diagnostics data. The tags for every column are loaded as indices.
        They will be converted to bools in preprocess_split function"""

        def _build_label_vocab(labelspace, data):
            """ This function builds the vocab mapping for a particular labelspace,
            which in practice is a coarse-grained diagnostic category. """
            values = set([d[labelspace] for d in data if labelspace in d])
            vocab = vocabulary.Vocabulary(counter=None, non_padded_namespaces=[labelspace])
            for value in values:
                vocab.add_token_to_namespace(value, labelspace)
            idx_to_word = vocab.get_index_to_token_vocabulary(labelspace)
            word_to_idx = vocab.get_token_to_index_vocabulary(labelspace)
            indexed = [[word_to_idx[d[labelspace]]] if labelspace in d else [] for d in data]
            return word_to_idx, idx_to_word, indexed

        def create_score_function(scorer, arg_to_scorer, tags_dict, tag_group):
            """ Create a scorer for each tag in a given tag group.
            The entire tag_group also has its own scorer"""
            setattr(self, "scorer__%s" % tag_group, scorer(arg_to_scorer))
            for index, tag in tags_dict.items():
                # 0 is missing value
                if index == 0:
                    continue
                setattr(self, "scorer__%s__%s" % (tag_group, tag), scorer(arg_to_scorer))

        targ_map = {"entailment": 1, "not_entailment": 0}
        data = [json.loads(d) for d in open(os.path.join(self.path, "AX-b.jsonl"))]
        sent1s = [
            tokenize_and_truncate(self._tokenizer_name, d["sentence1"], self.max_seq_len)
            for d in data
        ]
        sent2s = [
            tokenize_and_truncate(self._tokenizer_name, d["sentence2"], self.max_seq_len)
            for d in data
        ]
        labels = [targ_map[d["label"]] for d in data]
        idxs = [int(d["idx"]) for d in data]
        lxs2idx, idx2lxs, lxs = _build_label_vocab("lexical-semantics", data)
        pas2idx, idx2pas, pas = _build_label_vocab("predicate-argument-structure", data)
        lgc2idx, idx2lgc, lgc = _build_label_vocab("logic", data)
        knw2idx, idx2knw, knw = _build_label_vocab("knowledge", data)
        diag_data_dic = {
            "sents1": sent1s,
            "sents2": sent2s,
            "targs": labels,
            "idxs": idxs,
            "lex_sem": lxs,
            "pr_ar_str": pas,
            "logic": lgc,
            "knowledge": knw,
            "ix_to_lex_sem_dic": idx2lxs,
            "ix_to_pr_ar_str_dic": idx2pas,
            "ix_to_logic_dic": idx2lgc,
            "ix_to_knowledge_dic": idx2knw,
        }

        self.ix_to_lex_sem_dic = diag_data_dic["ix_to_lex_sem_dic"]
        self.ix_to_pr_ar_str_dic = diag_data_dic["ix_to_pr_ar_str_dic"]
        self.ix_to_logic_dic = diag_data_dic["ix_to_logic_dic"]
        self.ix_to_knowledge_dic = diag_data_dic["ix_to_knowledge_dic"]

        # Train, val, test splits are same. We only need one split but the code
        # probably expects all splits to be present.
        self.train_data_text = (
            diag_data_dic["sents1"],
            diag_data_dic["sents2"],
            diag_data_dic["targs"],
            diag_data_dic["idxs"],
            diag_data_dic["lex_sem"],
            diag_data_dic["pr_ar_str"],
            diag_data_dic["logic"],
            diag_data_dic["knowledge"],
        )
        self.val_data_text = self.train_data_text
        self.test_data_text = self.train_data_text
        self.sentences = self.train_data_text[0] + self.train_data_text[1]
        log.info("\tFinished loading diagnostic data.")

        # TODO: use FastMatthews instead to save memory.
        create_score_function(Correlation, "matthews", self.ix_to_lex_sem_dic, "lex_sem")
        create_score_function(Correlation, "matthews", self.ix_to_pr_ar_str_dic, "pr_ar_str")
        create_score_function(Correlation, "matthews", self.ix_to_logic_dic, "logic")
        create_score_function(Correlation, "matthews", self.ix_to_knowledge_dic, "knowledge")
        self._scorer_all_mcc = Correlation("matthews")  # score all examples according to MCC
        self._scorer_all_acc = CategoricalAccuracy()  # score all examples according to acc
        log.info("\tFinished creating score functions for diagnostic data.")


@register_task("winogender-diagnostic", rel_path="RTE/diagnostics", n_classes=2)
class WinogenderTask(GLUEDiagnosticTask):
    """Supported winogender task """

    def __init__(self, path, max_seq_len, name, n_classes, **kw):
        super().__init__(path, max_seq_len, name, n_classes, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.n_classes = n_classes

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None
        self.acc_scorer = BooleanAccuracy()
        self.gender_parity_scorer = GenderParity()
        self.val_metric = "%s_accuracy" % name

    def load_data(self):
        """ Process the datasets located at path. """
        targ_map = {"not_entailment": 0, "entailment": 1}

        self.train_data_text = load_pair_nli_jsonl(
            os.path.join(self.path, "AX-g.jsonl"), self._tokenizer_name, self.max_seq_len, targ_map
        )
        self.val_data_text = load_pair_nli_jsonl(
            os.path.join(self.path, "AX-g.jsonl"), self._tokenizer_name, self.max_seq_len, targ_map
        )
        self.test_data_text = load_pair_nli_jsonl(
            os.path.join(self.path, "AX-g.jsonl"), self._tokenizer_name, self.max_seq_len, targ_map
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading winogender (from SuperGLUE formatted data).")

    def process_split(self, split, indexers, model_preprocessing_interface):
        def _make_instance(input1, input2, labels, idx, pair_id):
            d = {}
            d["sent1_str"] = MetadataField(" ".join(input1))
            if model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                inp = model_preprocessing_interface.boundary_token_fn(input1, input2)
                d["inputs"] = sentence_to_text_field(inp, indexers)
                d["sent2_str"] = MetadataField(" ".join(input2))
            else:
                d["input1"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(input1), indexers
                )
                if input2:
                    d["input2"] = sentence_to_text_field(
                        model_preprocessing_interface.boundary_token_fn(input2), indexers
                    )
                    d["sent2_str"] = MetadataField(" ".join(input2))
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idxs_tags", skip_indexing=True)
            d["pair_id"] = LabelField(pair_id, label_namespace="pair_id_tags", skip_indexing=True)
            return Instance(d)

        instances = map(_make_instance, *split)
        return instances  # lazy iterator

    def get_metrics(self, reset=False):
        return {
            "accuracy": self.acc_scorer.get_metric(reset),
            "gender_parity": self.gender_parity_scorer.get_metric(reset),
        }

    def update_diagnostic_metrics(self, logits, labels, batch):
        _, preds = logits.max(dim=1)
        self.acc_scorer(preds, labels)
        batch["preds"] = preds
        # Convert batch to dict to fit gender_parity_scorer API.
        batch_dict = [
            {
                key: batch[key][index]
                for key in batch.keys()
                if key not in ["input1", "input2", "inputs"]
            }
            for index in range(len(batch["sent1_str"]))
        ]
        self.gender_parity_scorer(batch_dict)


@register_task("rte", rel_path="RTE/")
class RTETask(PairClassificationTask):
    """ Task class for Recognizing Textual Entailment 1, 2, 3, 5 """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        """ Process the datasets located at path. """
        targ_map = {"not_entailment": 0, "entailment": 1}
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading RTE (from GLUE formatted data).")


@register_task("rte-superglue", rel_path="RTE/")
class RTESuperGLUETask(RTETask):
    """ Task class for Recognizing Textual Entailment 1, 2, 3, 5
    Uses JSONL format used by SuperGLUE"""

    def load_data(self):
        """ Process the datasets located at path. """
        targ_map = {"not_entailment": 0, True: 1, False: 0, "entailment": 1}

        def _load_jsonl(data_file):
            data = [json.loads(d) for d in open(data_file, encoding="utf-8")]
            sent1s, sent2s, trgs, idxs = [], [], [], []
            for example in data:
                sent1s.append(
                    tokenize_and_truncate(
                        self._tokenizer_name, example["premise"], self.max_seq_len
                    )
                )
                sent2s.append(
                    tokenize_and_truncate(
                        self._tokenizer_name, example["hypothesis"], self.max_seq_len
                    )
                )
                trg = targ_map[example["label"]] if "label" in example else 0
                trgs.append(trg)
                idxs.append(example["idx"])
            return [sent1s, sent2s, trgs, idxs]

        self.train_data_text = _load_jsonl(os.path.join(self.path, "train.jsonl"))
        self.val_data_text = _load_jsonl(os.path.join(self.path, "val.jsonl"))
        self.test_data_text = _load_jsonl(os.path.join(self.path, "test.jsonl"))

        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading RTE (from SuperGLUE formatted data).")


@register_task("qnli", rel_path="QNLI/")
# second copy for different params
@register_task("qnli-alt", rel_path="QNLI/")
class QNLITask(PairClassificationTask):
    """Task class for SQuAD NLI"""

    def __init__(self, path, max_seq_len, name, **kw):
        super(QNLITask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        """Load the data"""
        targ_map = {"not_entailment": 0, "entailment": 1}
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading QNLI.")


@register_task("wnli", rel_path="WNLI/")
class WNLITask(PairClassificationTask):
    """Class for Winograd NLI task"""

    def __init__(self, path, max_seq_len, name, **kw):
        super(WNLITask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        """Load the data"""
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading Winograd.")


@register_task("joci", rel_path="JOCI/")
class JOCITask(PairOrdinalRegressionTask):
    """Class for JOCI ordinal regression task"""

    def __init__(self, path, max_seq_len, name, **kw):
        super(JOCITask, self).__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            skip_rows=1,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            skip_rows=1,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            skip_rows=1,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading JOCI data.")


@register_task("wiki103-classif", rel_path="WikiText103/")
class Wiki103Classification(PairClassificationTask):
    """Pair Classificaiton Task using Wiki103"""

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
        self.scorer2 = None
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.files_by_split = {
            "train": os.path.join(path, "train.sentences.txt"),
            "val": os.path.join(path, "valid.sentences.txt"),
            "test": os.path.join(path, "test.sentences.txt"),
        }
        self.max_seq_len = max_seq_len
        self.min_seq_len = 0

    def load_data(self):
        # Data is exposed as iterable: no preloading
        pass

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.
        Split should be one of 'train', 'val', or 'test'.
        """
        return self.load_data_for_path(self.files_by_split[split])

    def load_data_for_path(self, path):
        """ Rather than return a whole list of examples, stream them
        See WikiTextLMTask for an explanation of the preproc"""
        nonatomics_toks = [UNK_TOK_ALLENNLP, "<unk>"]
        with open(path) as txt_fh:
            for row in txt_fh:
                toks = row.strip()
                if not toks:
                    continue
                sent = atomic_tokenize(
                    toks,
                    UNK_TOK_ATOMIC,
                    nonatomics_toks,
                    self.max_seq_len,
                    tokenizer_name=self._tokenizer_name,
                )
                if sent.count("=") >= 2 or len(toks) < self.min_seq_len + 2:
                    continue
                yield sent

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for sent in self.load_data_for_path(path):
                yield sent

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process a language modeling split.  Split is a single list of sentences here.  """

        def _make_instance(input1, input2, labels):
            d = {}
            d["input1"] = sentence_to_text_field(
                model_preprocessing_interface.boundary_token_fn(input1), indexers
            )
            d["input2"] = sentence_to_text_field(
                model_preprocessing_interface.boundary_token_fn(input2), indexers
            )
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
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
        """ Compute here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            # pair sentence # = sent # - 1
            example_counts[split] = sum(1 for line in open(split_path)) - 1
        self.example_counts = example_counts


# Task class for DisSent with Wikitext 103 only considering clauses from within a single sentence
# Data sets should be prepared as described in Nie, Bennett, and Goodman (2017)
@register_task("dissentwiki", rel_path="DisSent/wikitext/", prefix="wikitext.dissent.single_sent")
# Task class for DisSent with Wikitext 103 considering clauses from within a single sentence
# or across two sentences.
# Data sets should be prepared as described in Nie, Bennett, and Goodman (2017)
@register_task("dissentwikifullbig", rel_path="DisSent/wikitext/", prefix="wikitext.dissent.big")
class DisSentTask(PairClassificationTask):
    """ Task class for DisSent, dataset agnostic.
        Based on Nie, Bennett, and Goodman (2017), but with different datasets.
    """

    def __init__(self, path, max_seq_len, prefix, name, **kw):
        """ There are 8 classes because there are 8 discourse markers in
            the dataset (and, but, because, if, when, before, though, so)
        """
        super().__init__(name, n_classes=8, **kw)
        self.max_seq_len = max_seq_len
        self.files_by_split = {
            "train": os.path.join(path, "%s.train" % prefix),
            "val": os.path.join(path, "%s.valid" % prefix),
            "test": os.path.join(path, "%s.test" % prefix),
        }

    def load_data(self):
        # Data is exposed as iterable: no preloading
        pass

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.

        Split should be one of 'train', 'val', or 'test'.
        """
        return self.load_data_for_path(self.files_by_split[split])

    def load_data_for_path(self, path):
        """ Load data """
        with open(path, "r") as txt_fh:
            for row in txt_fh:
                row = row.strip().split("\t")
                if len(row) != 3 or not (row[0] and row[1] and row[2]):
                    continue
                sent1 = tokenize_and_truncate(self._tokenizer_name, row[0], self.max_seq_len)
                sent2 = tokenize_and_truncate(self._tokenizer_name, row[1], self.max_seq_len)
                targ = int(row[2])
                yield (sent1, sent2, targ)

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            """ Don't use test set for vocab building. """
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for sent1, sent2, _ in self.load_data_for_path(path):
                yield sent1
                yield sent2

    def count_examples(self):
        """ Compute the counts here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for line in open(split_path))
        self.example_counts = example_counts

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(input1, input2, labels):
            d = {}
            if model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                inp = model_preprocessing_interface.boundary_token_fn(
                    input1, input2
                )  # drop leading [CLS] token
                d["inputs"] = sentence_to_text_field(inp, indexers)
            else:
                d["input1"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(input1), indexers
                )
                d["input2"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(input2), indexers
                )
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            return Instance(d)

        for sent1, sent2, trg in split:
            yield _make_instance(sent1, sent2, trg)


@register_task("recast-puns", rel_path="DNC/recast_puns_data")
@register_task("recast-ner", rel_path="DNC/recast_ner_data")
@register_task("recast-verbnet", rel_path="DNC/recast_verbnet_data")
@register_task("recast-verbcorner", rel_path="DNC/recast_verbcorner_data")
@register_task("recast-sentiment", rel_path="DNC/recast_sentiment_data")
@register_task("recast-factuality", rel_path="DNC/recast_factuality_data")
@register_task("recast-winogender", rel_path="DNC/manually-recast-winogender")
@register_task("recast-lexicosyntax", rel_path="DNC/lexicosyntactic_recasted")
@register_task("recast-kg", rel_path="DNC/kg-relations")
class RecastNLITask(PairClassificationTask):
    """ Task class for NLI Recast Data"""

    def __init__(self, path, max_seq_len, name, **kw):
        super(RecastNLITask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "train.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            skip_rows=0,
            label_idx=3,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "dev.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=0,
            s2_idx=1,
            skip_rows=0,
            label_idx=3,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "test.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            skip_rows=0,
            label_idx=3,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading recast probing data.")


class TaggingTask(Task):
    """ Generic tagging task, one tag per word """

    def __init__(self, name, num_tags, **kw):
        super().__init__(name, **kw)
        assert num_tags > 0
        self.num_tags = num_tags
        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.all_labels = [str(i) for i in range(self.num_tags)]
        self._label_namespace = self.name + "_tags"
        self.target_indexer = {"words": SingleIdTokenIndexer(namespace=self._label_namespace)}

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}

    def get_all_labels(self) -> List[str]:
        return self.all_labels

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = batch["labels"]
        assert len(self.get_scorers()) > 0, "Please specify a score metric"
        for scorer in self.get_scorers():
            scorer(logits, labels)


@register_task("ccg", rel_path="CCG/")
class CCGTaggingTask(TaggingTask):
    """ CCG supertagging as a task.
        Using the supertags from CCGbank. """

    def __init__(self, path, max_seq_len, name, tokenizer_name, **kw):
        """ There are 1363 supertags in CCGBank without introduced token. """
        from jiant.huggingface_transformers_interface import input_module_uses_transformers

        subword_tokenization = input_module_uses_transformers(tokenizer_name)
        super().__init__(
            name, 1363 + int(subword_tokenization), tokenizer_name=tokenizer_name, **kw
        )
        self.path = path
        self.INTRODUCED_TOKEN = "1363"
        self.subword_tokenization = subword_tokenization
        self.max_seq_len = max_seq_len
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def update_metrics(self, out, batch):
        logits, labels = out["logits"], out["labels"]
        for scorer in self.get_scorers():
            scorer(logits, labels)

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process a CCG tagging task """

        def _make_instance(input1, input2, target, mask):
            d = {}
            d["inputs"] = sentence_to_text_field(
                model_preprocessing_interface.boundary_token_fn(input1), indexers
            )
            d["sent1_str"] = MetadataField(" ".join(input1))
            d["targs"] = sentence_to_text_field(target, self.target_indexer)
            d["mask"] = MultiLabelField(
                mask, label_namespace="idx_tags", skip_indexing=True, num_labels=511
            )
            return Instance(d)

        split = list(split)
        split[1] = itertools.repeat(None)

        instances = map(_make_instance, *split)
        return instances

    def load_data(self):
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "ccg.train." + self._tokenizer_name),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            delimiter="\t",
            label_fn=lambda t: t.split(" "),
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "ccg.dev." + self._tokenizer_name),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            delimiter="\t",
            label_fn=lambda t: t.split(" "),
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "ccg.test." + self._tokenizer_name),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            delimiter="\t",
            has_labels=False,
        )

        # Get the mask for each sentence, where the mask is whether or not
        # the token was split off by tokenization. We want to only count the first
        # sub-piece in the BERT tokenization in the loss and score, following Devlin's NER
        # experiment
        # [BERT: Pretraining of Deep Bidirectional Transformers for Language Understanding]
        # (https://arxiv.org/abs/1810.04805)
        if self.subword_tokenization:
            import numpy.ma as ma

            masks = []
            for dataset in [tr_data, val_data]:
                dataset_mask = []
                for i in range(len(dataset[2])):
                    mask = ma.getmask(
                        ma.masked_where(
                            np.array(dataset[2][i]) != self.INTRODUCED_TOKEN,
                            np.array(dataset[2][i]),
                        )
                    )
                    mask_indices = np.where(mask)[0].tolist()
                    dataset_mask.append(mask_indices)
                masks.append(dataset_mask)

        # mock labels for test data (tagging)
        te_targs = [["0"] * len(x) for x in te_data[0]]
        te_mask = [list(range(len(x))) for x in te_data[0]]
        self.train_data_text = list(tr_data) + [masks[0]]
        self.val_data_text = list(val_data) + [masks[1]]
        self.test_data_text = list(te_data[:2]) + [te_targs] + [te_mask]
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        log.info("\tFinished loading CCGTagging data.")


class SpanClassificationTask(Task):
    """
    Generic class for span tasks.
    Acts as a classifier, but with multiple targets for each input text.
    Targets are of the form (span1, span2,..., span_n, label), where the spans are
    half-open token intervals [i, j).
    The number of spans is constant across examples.
    """

    def __init__(
        self,
        path: str,
        max_seq_len: int,
        name: str,
        label_file: str = None,
        files_by_split: Dict[str, str] = None,
        num_spans: int = 2,
        **kw,
    ):
        """
        Construct a span task.
        @register_task.

        Parameters
        ---------------------
            path: data directory
            max_seq_len: maximum sequence length (currently ignored)
            name: task name
            label_file: relative path to labels file
                - should be a line-delimited file where each line is a value the
                label can take.
            files_by_split: split name ('train', 'val', 'test') mapped to
                relative filenames (e.g. 'train': 'train.json')
        """
        super().__init__(name, **kw)

        assert label_file is not None
        assert files_by_split is not None
        self._files_by_split = {
            split: os.path.join(path, fname) for split, fname in files_by_split.items()
        }
        self.num_spans = num_spans
        self.max_seq_len = max_seq_len

        self._iters_by_split = None

        self.label_file = os.path.join(path, label_file)
        self.n_classes = None
        self._label_namespace = self.name + "_labels"

        self.acc_scorer = BooleanAccuracy()  # binary accuracy
        self.f1_scorer = F1Measure(positive_label=1)  # binary F1 overall
        self.scorers = [self.acc_scorer, self.f1_scorer]
        self.val_metric = "%s_f1" % self.name
        self.val_metric_decreases = False

    def _stream_records(self, filename):
        """
        Helper function for loading the data, which is in json format and
        checks if it has targets.
        """
        skip_ctr = 0
        total_ctr = 0
        for record in utils.load_json_data(filename):
            total_ctr += 1
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

    def get_split_text(self, split: str):
        """
        Get split text as iterable of records.
        Split should be one of 'train', 'val', or 'test'.
        """
        return self._iters_by_split[split]

    def get_num_examples(self, split_text):
        """
        Return number of examples in the result of get_split_text.
        Subclass can override this if data is not stored in column format.
        """
        return len(split_text)

    def _make_span_field(self, s, text_field, offset=1):
        # AllenNLP span extractor expects inclusive span indices
        # so minus 1 at the end index.
        return SpanField(s[0] + offset, s[1] - 1 + offset, text_field)

    def make_instance(self, record, idx, indexers, model_preprocessing_interface) -> Type[Instance]:
        """Convert a single record to an AllenNLP Instance."""
        tokens = record["text"].split()
        tokens, offset = model_preprocessing_interface.boundary_token_fn(tokens, get_offset=True)
        text_field = sentence_to_text_field(tokens, indexers)

        example = {}
        example["idx"] = MetadataField(idx)

        example["input1"] = text_field

        for i in range(self.num_spans):
            example["span" + str(i + 1) + "s"] = ListField(
                [self._make_span_field(record["target"]["span" + str(i + 1)], text_field, offset)]
            )
        example["labels"] = LabelField(
            record["label"], label_namespace="labels", skip_indexing=True
        )
        return Instance(example)

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
        metrics["acc"] = self.acc_scorer.get_metric(reset)
        precision, recall, f1 = self.f1_scorer.get_metric(reset)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        return metrics

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = batch["labels"]
        assert len(self.get_scorers()) > 0, "Please specify a score metric"
        for scorer in self.get_scorers():
            scorer(logits, labels)


@register_task("commitbank", rel_path="CB/")
class CommitmentTask(PairClassificationTask):
    """ NLI-formatted task detecting speaker commitment.
    Data and more info at github.com/mcdm/CommitmentBank/
    Paper forthcoming. """

    def __init__(self, path, max_seq_len, name, **kw):
        """ We use three F1 trackers, one for each class to compute multi-class F1 """
        super().__init__(name, n_classes=3, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer2 = F1Measure(0)
        self.scorer3 = F1Measure(1)
        self.scorer4 = F1Measure(2)
        self.scorers = [self.scorer1, self.scorer2, self.scorer3, self.scorer4]
        self.val_metric = "%s_f1" % name

    def load_data(self):
        """Process the dataset located at each data file.
           The target needs to be split into tokens because
           it is a sequence (one tag per input token). """
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}

        def _load_data(data_file):
            data = [json.loads(l) for l in open(data_file, encoding="utf-8").readlines()]
            sent1s, sent2s, targs, idxs = [], [], [], []
            for example in data:
                sent1s.append(
                    tokenize_and_truncate(
                        self._tokenizer_name, example["premise"], self.max_seq_len
                    )
                )
                sent2s.append(
                    tokenize_and_truncate(
                        self._tokenizer_name, example["hypothesis"], self.max_seq_len
                    )
                )
                trg = targ_map[example["label"]] if "label" in example else 0
                targs.append(trg)
                idxs.append(example["idx"])
            assert len(sent1s) == len(sent2s) == len(targs) == len(idxs), "Error processing CB data"
            return [sent1s, sent2s, targs, idxs]

        self.train_data_text = _load_data(os.path.join(self.path, "train.jsonl"))
        self.val_data_text = _load_data(os.path.join(self.path, "val.jsonl"))
        self.test_data_text = _load_data(os.path.join(self.path, "test.jsonl"))
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[1]
        )

        log.info("\tFinished loading CommitmentBank data.")

    def get_metrics(self, reset=False):
        """Get metrics specific to the task.
            - scorer1 tracks accuracy
            - scorers{2,3,4} compute class-specific F1,
                and we macro-average to get multi-class F1"""
        acc = self.scorer1.get_metric(reset)
        pcs1, rcl1, f11 = self.scorer2.get_metric(reset)
        pcs2, rcl2, f12 = self.scorer3.get_metric(reset)
        pcs3, rcl3, f13 = self.scorer4.get_metric(reset)
        pcs = (pcs1 + pcs2 + pcs3) / 3
        rcl = (rcl1 + rcl2 + rcl3) / 3
        f1 = (f11 + f12 + f13) / 3
        return {"accuracy": acc, "f1": f1, "precision": pcs, "recall": rcl}


@register_task("wic", rel_path="WiC/")
class WiCTask(PairClassificationTask):
    """ Task class for Words in Context. """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = CategoricalAccuracy()
        self.scorer2 = F1Measure(1)
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False

    def load_data(self):
        """Process the dataset located at data_file."""

        trg_map = {"true": 1, "false": 0, True: 1, False: 0}

        aligner_fn = get_aligner_fn(self._tokenizer_name)

        def _process_preserving_word(sent, word):
            """ Find out the index of the [first] instance of the word in the original sentence,
            and project the span containing marked word to the span containing tokens created from
            the marked word. """
            token_aligner, sent_tok = aligner_fn(sent)
            raw_start_idx = len(sent.split(word)[0].split(" ")) - 1
            # after spliting, there could be three cases, 1. a tailing space, 2. characters in front
            # of the keyword, 3. the sentence starts with the keyword
            raw_end_idx = len(word.split()) + raw_start_idx
            start_idx, end_idx = token_aligner.project_span(raw_start_idx, raw_end_idx)
            assert end_idx > start_idx, "Invalid marked word indices. Something is wrong."
            return sent_tok, start_idx, end_idx

        def _load_split(data_file):
            sents1, sents2, idxs1, idxs2, trgs, idxs = [], [], [], [], [], []
            with open(data_file, "r") as data_fh:
                for row in data_fh:
                    row = json.loads(row)
                    sent1 = row["sentence1"]
                    sent2 = row["sentence2"]
                    word1 = sent1[row["start1"] : row["end1"]]
                    word2 = sent2[row["start2"] : row["end2"]]
                    sent1, start1, end1 = _process_preserving_word(sent1, word1)
                    sent2, start2, end2 = _process_preserving_word(sent2, word2)
                    sents1.append(sent1)
                    sents2.append(sent2)
                    idxs1.append((start1, end1))
                    idxs2.append((start2, end2))
                    trg = trg_map[row["label"]] if "label" in row else 0
                    trgs.append(trg)
                    idxs.append(row["idx"])
                    assert (
                        "version" in row and row["version"] == 1.1
                    ), "WiC version is not v1.1; examples indices are likely incorrect and data "
                    "is likely pre-tokenized. Please re-download the data from "
                    "super.gluebenchmark.com."
                return [sents1, sents2, idxs1, idxs2, trgs, idxs]

        self.train_data_text = _load_split(os.path.join(self.path, "train.jsonl"))
        self.val_data_text = _load_split(os.path.join(self.path, "val.jsonl"))
        self.test_data_text = _load_split(os.path.join(self.path, "test.jsonl"))
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading WiC data.")

    def process_split(self, split, indexers, model_preprocessing_interface):
        """
        Convert a dataset of sentences into padded sequences of indices. Shared
        across several classes.

        """
        # check here if using bert to avoid passing model info to tasks

        def _make_instance(input1, input2, idxs1, idxs2, labels, idx):
            d = {}
            d["sent1_str"] = MetadataField(" ".join(input1))
            d["sent2_str"] = MetadataField(" ".join(input2))
            if model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                inp, offset1, offset2 = model_preprocessing_interface.boundary_token_fn(
                    input1, input2, get_offset=True
                )
                d["inputs"] = sentence_to_text_field(inp[: self.max_seq_len], indexers)
            else:
                inp1, offset1 = model_preprocessing_interface.boundary_token_fn(
                    input1, get_offset=True
                )
                inp2, offset2 = model_preprocessing_interface.boundary_token_fn(
                    input2, get_offset=True
                )
                d["input1"] = sentence_to_text_field(inp1[: self.max_seq_len], indexers)
                d["input2"] = sentence_to_text_field(inp2[: self.max_seq_len], indexers)
            d["idx1"] = ListField(
                [
                    NumericField(i)
                    for i in range(
                        min(idxs1[0] + offset1, self.max_seq_len - 1),
                        min(idxs1[1] + offset1, self.max_seq_len),
                    )
                ]
            )
            d["idx2"] = ListField(
                [
                    NumericField(i)
                    for i in range(
                        min(idxs2[0] + offset2, self.max_seq_len - 1),
                        min(idxs2[1] + offset2, self.max_seq_len),
                    )
                ]
            )
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idxs_tags", skip_indexing=True)

            return Instance(d)

        # Map over columns: input1, (input2), labels, idx
        instances = map(_make_instance, *split)
        return instances  # lazy iterator

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        pcs, rcl, f1 = self.scorer2.get_metric(reset)
        return {"accuracy": acc, "f1": f1, "precision": pcs, "recall": rcl}


class MultipleChoiceTask(Task):
    """ Generic task class for a multiple choice
    where each example consists of a question and
    a (possibly variable) number of possible answers"""

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = batch["label"]
        assert len(self.get_scorers()) > 0, "Please specify a score metric"
        for scorer in self.get_scorers():
            scorer(logits, labels)


@register_task("socialiqa", rel_path="SocialIQA/")
class SocialIQATask(MultipleChoiceTask):
    """ Task class for SocialIQA.
    Paper: https://homes.cs.washington.edu/~msap/pdfs/sap2019socialIQa.pdf
    Website and data: https://maartensap.github.io/social-iqa/
    """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False
        self.n_choices = 3
        self._label_namespace = self.name + "_tags"

    def get_all_labels(self):
        return ["A", "B", "C", "D"]

    def load_data(self):
        """ Process the dataset located at path.  """

        def _load_split(data_file):
            contexts, questions, choices, targs = [], [], [], []
            data = [json.loads(l) for l in open(data_file, encoding="utf-8")]
            for example in data:
                context = example["context"]
                choice1 = example["answerA"]
                choice2 = example["answerB"]
                choice3 = example["answerC"]
                question = example["question"]
                choice = [
                    tokenize_and_truncate(self._tokenizer_name, choice, self.max_seq_len)
                    for choice in [choice1, choice2, choice3]
                ]
                targ = example["correct"] if "correct" in example else 0
                contexts.append(
                    tokenize_and_truncate(self._tokenizer_name, context, self.max_seq_len)
                )
                choices.append(choice)
                questions.append(
                    tokenize_and_truncate(self._tokenizer_name, question, self.max_seq_len)
                )
                targs.append(targ)
            return [contexts, choices, questions, targs]

        self.train_data_text = _load_split(os.path.join(self.path, "socialIQa_v1.4_trn.jsonl"))
        self.val_data_text = _load_split(os.path.join(self.path, "socialIQa_v1.4_dev.jsonl"))
        self.test_data_text = _load_split(os.path.join(self.path, "socialIQa_v1.4_tst.jsonl"))
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + [choice for choices in self.train_data_text[1] for choice in choices]
            + [choice for choices in self.val_data_text[1] for choice in choices]
        )
        log.info("\tFinished loading SocialIQA data.")

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(context, choices, question, label, idx):
            d = {}
            d["question_str"] = MetadataField(" ".join(context))
            if not model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(context), indexers
                )
            for choice_idx, choice in enumerate(choices):
                inp = (
                    model_preprocessing_interface.boundary_token_fn(context, question + choice)
                    if model_preprocessing_interface.model_flags["uses_pair_embedding"]
                    else model_preprocessing_interface.boundary_token_fn(choice)
                )
                d["choice%d" % choice_idx] = sentence_to_text_field(inp, indexers)
                d["choice%d_str" % choice_idx] = MetadataField(" ".join(choice))
            d["label"] = LabelField(label, label_namespace=self._label_namespace)
            d["idx"] = LabelField(idx, label_namespace="idxs_tags", skip_indexing=True)
            return Instance(d)

        split = list(split)
        if len(split) < 5:
            split.append(itertools.count())
        instances = map(_make_instance, *split)
        return instances

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}


class SpanPredictionTask(Task):
    """ Generic task class for predicting a span """

    n_classes = 2

    def update_metrics(self, out, batch):
        batch_size = len(out["logits"]["span_start"])
        logits_dict = out["logits"]
        pred_span_start = torch.argmax(logits_dict["span_start"], dim=1).cpu().numpy()
        pred_span_end = torch.argmax(logits_dict["span_end"], dim=1).cpu().numpy()

        pred_str_list = self.get_pred_str(
            out["logits"], batch, batch_size, pred_span_start, pred_span_end
        )
        gold_str_list = batch["answer_str"]

        """ A batch of logits+answer strings and the questions they go with """
        self.f1_metric(pred_str_list=pred_str_list, gold_str_list=gold_str_list)
        self.em_metric(pred_str_list=pred_str_list, gold_str_list=gold_str_list)

    def get_pred_str(self, preds, batch, batch_size, pred_span_start, pred_span_end):
        """
        For span prediction, we compute metrics based on span strings. This function
        gets the span string based on start and end index predictions.

        """
        pred_str_list = []
        for i in range(batch_size):

            # Adjust for start_offset (e.g. [CLS] tokens).
            pred_span_start_i = pred_span_start[i] - batch["start_offset"][i]
            pred_span_end_i = pred_span_end[i] - batch["start_offset"][i]

            # Ensure that predictions fit within the range of valid tokens
            pred_span_start_i = min(
                pred_span_start_i, len(batch["space_processed_token_map"][i]) - 1
            )
            pred_span_end_i = min(
                max(pred_span_end_i, pred_span_start_i + 1),
                len(batch["space_processed_token_map"][i]) - 1,
            )

            # space_processed_token_map is a list of tuples
            #   (space_token, processed_token (e.g. BERT), space_token_index)
            # The assumption is that each space_token corresponds to multiple processed_tokens.
            # After we get the corresponding start/end space_token_indices, we can do " ".join
            #   to get the corresponding string that is definitely within the original input.
            # One constraint here is that our predictions can only go up to a the granularity of
            # space_tokens. This is not so bad because SQuAD-style scripts also remove punctuation.
            pred_char_span_start = batch["space_processed_token_map"][i][pred_span_start_i][2]
            pred_char_span_end = batch["space_processed_token_map"][i][pred_span_end_i][2]
            pred_str_list.append(
                " ".join(
                    batch["passage_str"][i].split()[pred_char_span_start:pred_char_span_end]
                ).strip()
            )
        return pred_str_list

    def handle_preds(self, preds, batch):
        batch_size = len(preds["span_start"])
        preds["span_str"] = self.get_pred_str(
            preds, batch, batch_size, preds["span_start"], preds["span_end"]
        )
        return preds


@register_task("copa", rel_path="COPA/")
class COPATask(MultipleChoiceTask):
    """ Task class for Choice of Plausible Alternatives Task.  """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False
        self.n_choices = 2

    def load_data(self):
        """ Process the dataset located at path.  """

        def _load_split(data_file):
            contexts, questions, choicess, targs = [], [], [], []
            data = [json.loads(l) for l in open(data_file, encoding="utf-8")]
            for example in data:
                context = example["premise"]
                choice1 = example["choice1"]
                choice2 = example["choice2"]
                question = example["question"]
                question = (
                    "What was the cause of this?"
                    if question == "cause"
                    else "What happened as a result?"
                )
                choices = [
                    tokenize_and_truncate(self._tokenizer_name, choice, self.max_seq_len)
                    for choice in [choice1, choice2]
                ]
                targ = example["label"] if "label" in example else 0
                contexts.append(
                    tokenize_and_truncate(self._tokenizer_name, context, self.max_seq_len)
                )
                choicess.append(choices)
                questions.append(
                    tokenize_and_truncate(self._tokenizer_name, question, self.max_seq_len)
                )
                targs.append(targ)
            return [contexts, choicess, questions, targs]

        self.train_data_text = _load_split(os.path.join(self.path, "train.jsonl"))
        self.val_data_text = _load_split(os.path.join(self.path, "val.jsonl"))
        self.test_data_text = _load_split(os.path.join(self.path, "test.jsonl"))
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + [choice for choices in self.train_data_text[1] for choice in choices]
            + [choice for choices in self.val_data_text[1] for choice in choices]
        )
        log.info("\tFinished loading COPA (as QA) data.")

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(context, choices, question, label, idx):
            d = {}
            d["question_str"] = MetadataField(" ".join(context))
            if not model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(context), indexers
                )
            for choice_idx, choice in enumerate(choices):
                inp = (
                    model_preprocessing_interface.boundary_token_fn(context, question + choice)
                    if model_preprocessing_interface.model_flags["uses_pair_embedding"]
                    else model_preprocessing_interface.boundary_token_fn(choice)
                )
                d["choice%d" % choice_idx] = sentence_to_text_field(inp, indexers)
                d["choice%d_str" % choice_idx] = MetadataField(" ".join(choice))
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idxs_tags", skip_indexing=True)
            return Instance(d)

        split = list(split)
        if len(split) < 5:
            split.append(itertools.count())
        instances = map(_make_instance, *split)
        return instances

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}


@register_task("swag", rel_path="SWAG/")
class SWAGTask(MultipleChoiceTask):
    """ Task class for Situations with Adversarial Generations.  """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False
        self.n_choices = 4

    def load_data(self):
        """ Process the dataset located at path.  """

        def _load_split(data_file):
            questions, choicess, targs = [], [], []
            data = pd.read_csv(data_file)
            for ex_idx, ex in data.iterrows():
                sent1 = tokenize_and_truncate(self._tokenizer_name, ex["sent1"], self.max_seq_len)
                questions.append(sent1)
                sent2_prefix = ex["sent2"]
                choices = []
                for i in range(4):
                    choice = sent2_prefix + " " + ex["ending%d" % i]
                    choice = tokenize_and_truncate(self._tokenizer_name, choice, self.max_seq_len)
                    choices.append(choice)
                choicess.append(choices)
                targ = ex["label"] if "label" in ex else 0
                targs.append(targ)
            return [questions, choicess, targs]

        self.train_data_text = _load_split(os.path.join(self.path, "train.csv"))
        self.val_data_text = _load_split(os.path.join(self.path, "val.csv"))
        self.test_data_text = _load_split(os.path.join(self.path, "test.csv"))
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + [choice for choices in self.train_data_text[1] for choice in choices]
            + [choice for choices in self.val_data_text[1] for choice in choices]
        )
        log.info("\tFinished loading SWAG data.")

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(question, choices, label, idx):
            d = {}
            d["question_str"] = MetadataField(" ".join(question))
            if not model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(question), indexers
                )
            for choice_idx, choice in enumerate(choices):
                inp = (
                    model_preprocessing_interface.boundary_token_fn(question, choice)
                    if model_preprocessing_interface.model_flags["uses_pair_embedding"]
                    else model_preprocessing_interface.boundary_token_fn(choice)
                )
                d["choice%d" % choice_idx] = sentence_to_text_field(inp, indexers)
                d["choice%d_str" % choice_idx] = MetadataField(" ".join(choice))
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idxs_tags", skip_indexing=True)
            return Instance(d)

        split = list(split)
        if len(split) < 4:
            split.append(itertools.count())
        instances = map(_make_instance, *split)
        return instances

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}


@register_task("hellaswag", rel_path="HellaSwag/")
class HellaSwagTask(MultipleChoiceTask):
    """ Task class for HellaSwag.  """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False
        self.n_choices = 4

    def load_data(self):
        """ Process the dataset located at path.  """

        def _load_split(data_file):
            questions, choicess, targs, idxs = [], [], [], []
            data = [json.loads(l) for l in open(data_file, encoding="utf-8")]
            for example in data:
                sent1 = tokenize_and_truncate(
                    self._tokenizer_name, example["ctx_a"], self.max_seq_len
                )
                questions.append(sent1)
                sent2_prefix = example["ctx_b"]
                choices = [
                    tokenize_and_truncate(
                        self._tokenizer_name, sent2_prefix + " " + ending, self.max_seq_len
                    )
                    for ending in example["endings"]
                ]
                choicess.append(choices)
                targ = example["label"] if "label" in example else 0
                idx = example["ind"]
                targs.append(targ)
                idxs.append(idx)
            return [questions, choicess, targs, idxs]

        self.train_data_text = _load_split(os.path.join(self.path, "hellaswag_train.jsonl"))
        self.val_data_text = _load_split(os.path.join(self.path, "hellaswag_val.jsonl"))
        self.test_data_text = _load_split(os.path.join(self.path, "hellaswag_test.jsonl"))
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + [choice for choices in self.train_data_text[1] for choice in choices]
            + [choice for choices in self.val_data_text[1] for choice in choices]
        )
        log.info("\tFinished loading HellaSwag data.")

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(question, choices, label, idx):
            d = {}
            d["question_str"] = MetadataField(" ".join(question))
            if not model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(question), indexers
                )
            for choice_idx, choice in enumerate(choices):
                inp = (
                    model_preprocessing_interface.boundary_token_fn(question, choice)
                    if model_preprocessing_interface.model_flags["uses_pair_embedding"]
                    else model_preprocessing_interface.boundary_token_fn(choice)
                )
                d["choice%d" % choice_idx] = sentence_to_text_field(inp, indexers)
                d["choice%d_str" % choice_idx] = MetadataField(" ".join(choice))
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idxs_tags", skip_indexing=True)
            return Instance(d)

        split = list(split)
        instances = map(_make_instance, *split)
        return instances

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}


@register_task("winograd-coreference", rel_path="WSC")
class WinogradCoreferenceTask(SpanClassificationTask):
    def __init__(self, path, **kw):
        self._files_by_split = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
        self.num_spans = 2
        super().__init__(
            files_by_split=self._files_by_split, label_file="labels.txt", path=path, **kw
        )
        self.n_classes = 2
        self.val_metric = "%s_acc" % self.name

    def load_data(self):
        iters_by_split = collections.OrderedDict()
        for split, filename in self._files_by_split.items():
            if filename.endswith("test.jsonl"):
                iters_by_split[split] = load_span_data(
                    self.tokenizer_name, filename, has_labels=False
                )
            else:
                iters_by_split[split] = load_span_data(
                    self.tokenizer_name, filename, label_fn=lambda x: int(x)
                )
        self._iters_by_split = iters_by_split

    def update_metrics(self, out, batch):
        logits = out["logits"]
        labels = batch["labels"]
        logits, labels = logits.detach(), labels.detach()
        _, preds = logits.max(dim=1)

        self.f1_scorer(logits, labels)  # expects [batch_size, n_classes] for preds
        self.acc_scorer(preds, labels)  # expects [batch_size, 1] for both

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        collected_metrics = {
            "f1": self.f1_scorer.get_metric(reset)[2],
            "acc": self.acc_scorer.get_metric(reset),
        }
        return collected_metrics


@register_task("boolq", rel_path="BoolQ")
class BooleanQuestionTask(PairClassificationTask):
    """Task class for Boolean Questions Task."""

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer2 = F1Measure(1)
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_acc_f1" % name
        self.val_metric_decreases = False

    def load_data(self):
        """ Process the dataset located at path.  """

        def _load_jsonl(data_file):
            raw_data = [json.loads(d) for d in open(data_file, encoding="utf-8")]
            data = []
            for d in raw_data:
                question = tokenize_and_truncate(
                    self._tokenizer_name, d["question"], self.max_seq_len
                )
                passage = tokenize_and_truncate(
                    self._tokenizer_name, d["passage"], self.max_seq_len
                )
                new_datum = {"question": question, "passage": passage}
                answer = d["label"] if "label" in d else False
                new_datum["label"] = answer
                data.append(new_datum)
            return data

        self.train_data_text = _load_jsonl(os.path.join(self.path, "train.jsonl"))
        self.val_data_text = _load_jsonl(os.path.join(self.path, "val.jsonl"))
        self.test_data_text = _load_jsonl(os.path.join(self.path, "test.jsonl"))

        self.sentences = [d["question"] for d in self.train_data_text + self.val_data_text] + [
            d["passage"] for d in self.train_data_text + self.val_data_text
        ]
        log.info("\tFinished loading BoolQ data.")

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(d, idx):
            new_d = {}
            new_d["question_str"] = MetadataField(" ".join(d["question"]))
            new_d["passage_str"] = MetadataField(" ".join(d["passage"]))
            if not model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                new_d["input1"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(d["passage"]), indexers
                )
                new_d["input2"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(d["question"]), indexers
                )
            else:  # BERT/XLNet
                psg_qst = model_preprocessing_interface.boundary_token_fn(
                    d["passage"], d["question"]
                )
                new_d["inputs"] = sentence_to_text_field(psg_qst, indexers)
            new_d["labels"] = LabelField(d["label"], label_namespace="labels", skip_indexing=True)
            new_d["idx"] = LabelField(idx, label_namespace="idxs_tags", skip_indexing=True)
            return Instance(new_d)

        split = [split, itertools.count()]
        instances = map(_make_instance, *split)
        return instances

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        pcs, rcl, f1 = self.scorer2.get_metric(reset)
        return {
            "acc_f1": (acc + f1) / 2,
            "accuracy": acc,
            "f1": f1,
            "precision": pcs,
            "recall": rcl,
        }

    def count_examples(self, splits=["train", "val", "test"]):
        """ Count examples in the dataset. """
        self.example_counts = {}
        for split in splits:
            st = self.get_split_text(split)
            self.example_counts[split] = len(st)


@register_task("anli", rel_path="aNLI")
class AlphaNLITask(MultipleChoiceTask):
    """
    Task class for Abductive Natural Language Inference.

    Paper: https://arxiv.org/abs/1908.05739
    Website: https://leaderboard.allenai.org/anli/submissions/get-started
    """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False
        self.n_choices = 2

    def load_data(self):
        """ Process the dataset located at path.  """

        def _load_split(inputs_file, labels_file):
            obs1, hyp1, hyp2, obs2 = [], [], [], []
            with open(inputs_file, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    obs1.append(
                        tokenize_and_truncate(self._tokenizer_name, row["obs1"], self.max_seq_len)
                    )
                    hyp1.append(
                        tokenize_and_truncate(self._tokenizer_name, row["hyp1"], self.max_seq_len)
                    )
                    hyp2.append(
                        tokenize_and_truncate(self._tokenizer_name, row["hyp2"], self.max_seq_len)
                    )
                    obs2.append(
                        tokenize_and_truncate(self._tokenizer_name, row["obs2"], self.max_seq_len)
                    )
            with open(labels_file) as f:
                labels = [int(i) - 1 for i in f.read().split()]  # -1 to get {0, 1} labels
            return [obs1, hyp1, hyp2, obs2, labels]

        self.train_data_text = _load_split(
            inputs_file=os.path.join(self.path, "train.jsonl"),
            labels_file=os.path.join(self.path, "train-labels.lst"),
        )
        self.val_data_text = _load_split(
            inputs_file=os.path.join(self.path, "dev.jsonl"),
            labels_file=os.path.join(self.path, "dev-labels.lst"),
        )

        log.warning("aNLI has no public test set, so we reuse the dev set as a stand-in")
        self.test_data_text = self.val_data_text
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.train_data_text[2]
            + self.train_data_text[3]
            + self.val_data_text[0]
            + self.val_data_text[1]
            + self.val_data_text[2]
            + self.val_data_text[3]
        )
        log.info("\tFinished loading aNLI data.")

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(obs1, hyp1, hyp2, obs2, label, idx):
            d = {}
            if not model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                # We're combining obs1 and obs2 in a potentially suboptimal way here
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(obs1 + obs2), indexers
                )
                d["question_str"] = MetadataField(" ".join(obs1 + obs2))
                for hyp_idx, hyp in enumerate([hyp1, hyp2]):
                    d["choice%d" % hyp_idx] = sentence_to_text_field(
                        model_preprocessing_interface.boundary_token_fn(hyp), indexers
                    )
                    d["choice%d_str" % hyp_idx] = MetadataField(" ".join(hyp))
            else:
                for hyp_idx, hyp in enumerate([hyp1, hyp2]):
                    inp = (
                        model_preprocessing_interface.boundary_token_fn(obs1 + hyp, obs2)
                        if model_preprocessing_interface.model_flags["uses_pair_embedding"]
                        else model_preprocessing_interface.boundary_token_fn(hyp)
                    )
                    d["choice%d" % hyp_idx] = sentence_to_text_field(inp, indexers)
                    d["choice%d_str" % hyp_idx] = MetadataField(" ".join(inp))
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idxs_tags", skip_indexing=True)
            return Instance(d)

        split = list(split)
        if len(split) < 6:
            split.append(itertools.count())
        instances = map(_make_instance, *split)
        return instances

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}


@register_task("scitail", rel_path="SciTailV1.1/tsv_format/")
class SciTailTask(PairClassificationTask):
    """ Task class for SciTail http://data.allenai.org/scitail/ """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def load_data(self):
        """Process and load Scitail data"""
        targ_map = {"neutral": 0, "entails": 1}
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "scitail_1.0_train.tsv"),
            max_seq_len=self.max_seq_len,
            label_fn=targ_map.__getitem__,
        )
        self.val_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "scitail_1.0_dev.tsv"),
            max_seq_len=self.max_seq_len,
            label_fn=targ_map.__getitem__,
        )
        self.test_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "scitail_1.0_test.tsv"),
            max_seq_len=self.max_seq_len,
            label_fn=targ_map.__getitem__,
            return_indices=True,
        )
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        log.info("\tFinished loading SciTail")


@register_task("winogrande", rel_path="Winogrande/", train_size="xl")
# For experiment record management convenience, we use winogrande as an alias of winogrande-xl
@register_task("winogrande-xl", rel_path="Winogrande/", train_size="xl")
@register_task("winogrande-l", rel_path="Winogrande/", train_size="l")
@register_task("winogrande-m", rel_path="Winogrande/", train_size="m")
@register_task("winogrande-s", rel_path="Winogrande/", train_size="s")
@register_task("winogrande-xs", rel_path="Winogrande/", train_size="xs")
class WinograndeTask(MultipleChoiceTask):
    def __init__(self, path, max_seq_len, name, train_size, **kw):
        """
        Task class for Winogrande dataset.

        Paper: https://arxiv.org/abs/1907.10641
        Website (data download): https://mosaic.allenai.org/projects/winogrande
        Reference code: https://github.com/allenai/winogrande
        """
        super().__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.train_size = train_size
        self.n_choices = 2

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False

    def load_data(self):
        def _load_data(data_file):
            data = [json.loads(l) for l in open(data_file, encoding="utf-8").readlines()]
            contexts, choices, labels, idxs = [], [], [], []
            for i, example in enumerate(data):
                sent_part1, sent_part2 = example["sentence"].split("_")
                sent_part1_tokens = tokenize_and_truncate(
                    self._tokenizer_name, sent_part1, self.max_seq_len
                )
                choice_tokens = [
                    tokenize_and_truncate(
                        self._tokenizer_name, example["option1"] + sent_part2, self.max_seq_len
                    ),
                    tokenize_and_truncate(
                        self._tokenizer_name, example["option2"] + sent_part2, self.max_seq_len
                    ),
                ]
                contexts.append(sent_part1_tokens)
                choices.append(choice_tokens)
                labels.append(int(example["answer"] if "answer" in example else "1") - 1)
                idxs.append(example["qID"])
            return [contexts, choices, labels, idxs]

        self.train_data_text = _load_data(
            os.path.join(self.path, "train_%s.jsonl" % self.train_size)
        )
        self.val_data_text = _load_data(os.path.join(self.path, "dev.jsonl"))
        self.test_data_text = _load_data(os.path.join(self.path, "test.jsonl"))
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + self.train_data_text[1][0]
            + self.train_data_text[1][1]
            + self.val_data_text[1][0]
            + self.val_data_text[1][1]
        )

        log.info("\tFinished loading Winogrande data.")

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(context, choices, label, idx):
            # because Winogrande uses MultiTaskModel._mc_forward as its forward funcion, we adapt
            # to the keywords specified in _mc_forward, i.e. "question", "choice1" and "choice2".
            d = {}
            d["question_str"] = MetadataField(" ".join(context))
            if not model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(context), indexers
                )
            for choice_idx, choice in enumerate(choices):
                inp = (
                    model_preprocessing_interface.boundary_token_fn(context, choice)
                    if model_preprocessing_interface.model_flags["uses_pair_embedding"]
                    else model_preprocessing_interface.boundary_token_fn(choice)
                )
                d["choice%d" % choice_idx] = sentence_to_text_field(inp, indexers)
                d["choice%d_str" % choice_idx] = MetadataField(" ".join(choice))
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["idx"] = MetadataField(idx)
            return Instance(d)

        split = list(split)
        instances = map(_make_instance, *split)
        return instances

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}


@register_task("wikipedia_corpus_sop", rel_path="wikipedia_corpus_small")
class SentenceOrderTask(PairClassificationTask):
    """ Task class for Sentence Order Prediction (SOP). See the ALBERT paper for details on SOP:
        https://arxiv.org/abs/1909.11942.
        We are currently using an unpreprocessed version of the Wikipedia corpus
        (more specifically, the Wikidump data) that consists of 5% of the data. You can generate
        the data by following the instructions from jiant/scripts/mlm.
        One thing to note about our SOP ALBERT implementationn is that we do not load the pretrained
        weights for the SOP head beacuse they are unavailable in Huggingface. We only use the
        pretrained weights of the linear layer from ALBERT that creates the pooled output used in SOP.
    """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SentenceOrderTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None
        self.files_by_split = {
            "train": os.path.join(path, "train.txt"),
            "val": os.path.join(path, "valid.txt"),
            "test": os.path.join(path, "test.txt"),
        }
        self._label_namespace = self.name + "_labels"

    def get_data_iter(self, path):
        """Loading data file and tokenizing the text. We override the
        this function and all functions that call this function because
        the step of reading in the data for SOP is different than other
        PairClassificationTasks.

        Args:
            path: (str) data file path
        """
        f = open(path, "r")
        #  The dataset comes with one sentence per line, thus we split by
        #  line here.
        text = [line.strip() for line in f]
        for i in range(len(text) - 1):
            if random.uniform(0, 1) > 0.5:
                is_right_order = 1
                sent_a = text[i]
                sent_b = text[i + 1]
            else:
                is_right_order = 0
                sent_a = text[i + 1]
                sent_b = text[i]
            sent_a_processed = tokenize_and_truncate(
                self._tokenizer_name, sent_a, self.max_seq_len // 2
            )
            sent_b_processed = tokenize_and_truncate(
                self._tokenizer_name, sent_b, self.max_seq_len // 2
            )
            yield (sent_a_processed, sent_b_processed, is_right_order)

    def load_data(self):
        pass

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """Process a sentence order prediction split by indexing and creating fields.
        We override the PairClassificationTask process_split because our data split
        is different from the typical PairClassificationTask due to the more memory-efficient
        generator way of loading data we employ for SOP due to the dataset size.
        Args:
            split: (list) a single list of sentences
            indexers: (Indexer object) indexer to index input words
        """

        def _make_instance(sent_):
            sent_a, sent_b, is_right_order = sent_
            inp = model_preprocessing_interface.boundary_token_fn(sent_a, sent_b)
            input_sent = sentence_to_text_field(inp, indexers)
            label = LabelField(is_right_order, label_namespace="labels", skip_indexing=True)
            d = {"inputs": input_sent, "labels": label}
            return Instance(d)

        for sent in split:
            yield _make_instance(sent)

    def get_split_text(self, split: str):
        """Get split text as iterable of records.
        Args:
            split: (str) should be one of 'train', 'val', or 'test'.
        """
        return self.get_data_iter(self.files_by_split[split])

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """Yield sentences, used to compute vocabulary.
        """
        for split in self.files_by_split:
            # Don't use test set for vocab building.
            if split.startswith("test"):
                continue
            for sent in self.get_data_iter(self.files_by_split[split]):
                yield sent[0]
                yield sent[1]

    def count_examples(self):
        """Computes number of samples
        Assuming every line is one example.
        """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for _ in self.get_data_iter(split_path))
        self.example_counts = example_counts
