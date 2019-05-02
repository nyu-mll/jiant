import codecs
import collections
import copy
import itertools
import json
import logging as log
import os
from typing import Any, Dict, Iterable, List, Sequence, Type

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

from ..allennlp_mods.correlation import Correlation, FastMatthews
from ..allennlp_mods.numeric_field import NumericField
from ..utils import utils
from ..utils.data_loaders import get_tag_list, load_diagnostic_tsv, load_tsv, process_sentence
from ..utils.tokenizers import get_tokenizer
from ..utils.utils import truncate
from .registry import REGISTRY, register_task  # global task registry

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
    sent = process_sentence(tokenizer_name, sent, max_seq_len)
    sent = [nonatomic_toks[0] if t == atomic_tok else t for t in sent]
    return sent


def process_single_pair_task_split(split, indexers, is_pair=True, classification=True):
    """
    Convert a dataset of sentences into padded sequences of indices. Shared
    across several classes.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - indexers ()
        - is_pair (Bool)
        - classification (Bool)

    Returns:
        - instances (Iterable[Instance]): an iterable of AllenNLP Instances with fields
    """
    # check here if using bert to avoid passing model info to tasks
    is_using_bert = "bert_wpm_pretokenized" in indexers

    def _make_instance(input1, input2, labels, idx):
        d = {}
        d["sent1_str"] = MetadataField(" ".join(input1[1:-1]))
        if is_using_bert and is_pair:
            inp = input1 + input2[1:]  # throw away input2 leading [CLS]
            d["inputs"] = sentence_to_text_field(inp, indexers)
            d["sent2_str"] = MetadataField(" ".join(input2[1:-1]))
        else:
            d["input1"] = sentence_to_text_field(input1, indexers)
            if input2:
                d["input2"] = sentence_to_text_field(input2, indexers)
                d["sent2_str"] = MetadataField(" ".join(input2[1:-1]))
        if classification:
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
        else:
            d["labels"] = NumericField(labels)

        d["idx"] = LabelField(idx, label_namespace="idxs", skip_indexing=True)

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
        - truncate: truncate data to be at most some length
        - get_metrics:

    Outside the task:
        - process: pad and indexify data given a mapping
        - optimizer
    """

    def __init__(self, name, tokenizer_name):
        self.name = name
        self._tokenizer_name = tokenizer_name
        self.scorers = []

    def load_data(self, path, max_seq_len):
        """ Load data from path and create splits. """
        raise NotImplementedError

    def truncate(self, max_seq_len, sos_tok, eos_tok):
        """ Shorten sentences to max_seq_len and add sos and eos tokens. """
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

    def tokenizer_is_supported(self, tokenizer_name):
        """ Check if the tokenizer is supported for this task. """
        return get_tokenizer(tokenizer_name) is not None

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

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict:
        """ Get metrics specific to the task. """
        raise NotImplementedError

    def get_scorers(self):
        return self.scorers

    def update_metrics(self, logits, labels, tagmask=None):
        assert len(self.get_scorers()) > 0, "Please specify a score metric"
        for scorer in self.get_scorers():
            scorer(logits, labels)


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

    def truncate(self, max_seq_len, sos_tok="<SOS>", eos_tok="<EOS>"):
        self.train_data_text = [
            truncate(self.train_data_text[0], max_seq_len, sos_tok, eos_tok),
            self.train_data_text[1],
        ]
        self.val_data_text = [
            truncate(self.val_data_text[0], max_seq_len, sos_tok, eos_tok),
            self.val_data_text[1],
        ]
        self.test_data_text = [
            truncate(self.test_data_text[0], max_seq_len, sos_tok, eos_tok),
            self.test_data_text[1],
        ]

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(split, indexers, is_pair=False)


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

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(split, indexers, is_pair=True)


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

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(split, indexers, is_pair=True, classification=False)


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

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        return process_single_pair_task_split(split, indexers, is_pair=True, classification=False)

    def update_metrics():
        # currently don't support metrics for regression task
        # TODO(Yada): support them!
        return


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

    def update_metrics():
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
        """ """
        super(SSTTask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]

    def load_data(self, path, max_seq_len):
        """ Load data """
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            s1_idx=0,
            s2_idx=None,
            label_idx=1,
            skip_rows=1,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            s1_idx=0,
            s2_idx=None,
            label_idx=1,
            skip_rows=1,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=None,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SST data.")


@register_task("npi_adv_li", rel_path="NPI/probing/adverbs/licensor")
@register_task("npi_adv_sc", rel_path="NPI/probing/adverbs/scope_with_licensor")
@register_task("npi_adv_pr", rel_path="NPI/probing/adverbs/npi_present")
@register_task("npi_cond_li", rel_path="NPI/probing/conditionals/licensor")
@register_task("npi_cond_sc", rel_path="NPI/probing/conditionals/scope_with_licensor")
@register_task("npi_cond_pr", rel_path="NPI/probing/conditionals/npi_present")
@register_task("npi_negdet_li", rel_path="NPI/probing/determiner_negation_biclausal/licensor")
@register_task(
    "npi_negdet_sc", rel_path="NPI/probing/determiner_negation_biclausal/scope_with_licensor"
)
@register_task("npi_negdet_pr", rel_path="NPI/probing/determiner_negation_biclausal/npi_present")
@register_task("npi_negsent_li", rel_path="NPI/probing/sentential_negation_biclausal/licensor")
@register_task(
    "npi_negsent_sc", rel_path="NPI/probing/sentential_negation_biclausal/scope_with_licensor"
)
@register_task("npi_negsent_pr", rel_path="NPI/probing/sentential_negation_biclausal/npi_present")
@register_task("npi_only_li", rel_path="NPI/probing/only/licensor")
@register_task("npi_only_sc", rel_path="NPI/probing/only/scope_with_licensor")
@register_task("npi_only_pr", rel_path="NPI/probing/only/npi_present")
@register_task("npi_qnt_li", rel_path="NPI/probing/quantifiers/licensor")
@register_task("npi_qnt_sc", rel_path="NPI/probing/quantifiers/scope_with_licensor")
@register_task("npi_qnt_pr", rel_path="NPI/probing/quantifiers/npi_present")
@register_task("npi_ques_li", rel_path="NPI/probing/questions/licensor")
@register_task("npi_ques_sc", rel_path="NPI/probing/questions/scope_with_licensor")
@register_task("npi_ques_pr", rel_path="NPI/probing/questions/npi_present")
@register_task("npi_quessmp_li", rel_path="NPI/probing/simplequestions/licensor")
@register_task("npi_quessmp_sc", rel_path="NPI/probing/simplequestions/scope_with_licensor")
@register_task("npi_quessmp_pr", rel_path="NPI/probing/simplequestions/npi_present")
@register_task("npi_sup_li", rel_path="NPI/probing/superlative/licensor")
@register_task("npi_sup_sc", rel_path="NPI/probing/superlative/scope_with_licensor")
@register_task("npi_sup_pr", rel_path="NPI/probing/superlative/npi_present")
@register_task("cola_npi_adv", rel_path="NPI/splits/adverbs")
@register_task("cola_npi_cond", rel_path="NPI/splits/conditionals")
@register_task("cola_npi_negdet", rel_path="NPI/splits/determiner_negation_biclausal")
@register_task("cola_npi_negsent", rel_path="NPI/splits/sentential_negation_biclausal")
@register_task("cola_npi_only", rel_path="NPI/splits/only")
@register_task("cola_npi_ques", rel_path="NPI/splits/questions")
@register_task("cola_npi_quessmp", rel_path="NPI/splits/simplequestions")
@register_task("cola_npi_qnt", rel_path="NPI/splits/quantifiers")
@register_task("cola_npi_sup", rel_path="NPI/splits/superlative")
@register_task("all_cola_npi", rel_path="NPI/combs/all_env")
@register_task("hd_cola_npi_adv", rel_path="NPI/combs/minus_adverbs")
@register_task("hd_cola_npi_cond", rel_path="NPI/combs/minus_conditionals")
@register_task("hd_cola_npi_negdet", rel_path="NPI/combs/minus_determiner_negation_biclausal")
@register_task("hd_cola_npi_negsent", rel_path="NPI/combs/minus_sentential_negation_biclausal")
@register_task("hd_cola_npi_only", rel_path="NPI/combs/minus_only")
@register_task("hd_cola_npi_ques", rel_path="NPI/combs/minus_questions")
@register_task("hd_cola_npi_quessmp", rel_path="NPI/combs/minus_simplequestions")
@register_task("hd_cola_npi_qnt", rel_path="NPI/combs/minus_quantifiers")
@register_task("hd_cola_npi_sup", rel_path="NPI/combs/minus_superlative")
class CoLANPITask(SingleClassificationTask):
    """Class for NPI-related task; same with Warstdadt acceptability task but outputs labels for test-set
       Note: Used for an NYU seminar, data not yet public"""

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super(CoLANPITask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        # self.scorer1 = Average()
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]

    def load_data(self, path, max_seq_len):
        """Load the data"""
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test_full.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading NPI Data.")

    def get_metrics(self, reset=False):
        return {"mcc": self.scorer1.get_metric(reset), "accuracy": self.scorer2.get_metric(reset)}

    def update_metrics(self, logits, labels, tagmask=None):
        logits, labels = logits.detach(), labels.detach()
        _, preds = logits.max(dim=1)
        self.scorer1(preds, labels)
        self.scorer2(logits, labels)
        return


@register_task("cola", rel_path="CoLA/")
class CoLATask(SingleClassificationTask):
    """Class for Warstdadt acceptability task"""

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super(CoLATask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        # self.scorer1 = Average()
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]

    def load_data(self, path, max_seq_len):
        """Load the data"""
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=1,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=None,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading CoLA.")

    def get_metrics(self, reset=False):
        return {"mcc": self.scorer1.get_metric(reset), "accuracy": self.scorer2.get_metric(reset)}

    def update_metrics(self, logits, labels, tagmask=None):
        logits, labels = logits.detach(), labels.detach()
        _, preds = logits.max(dim=1)
        self.scorer1(preds, labels)
        self.scorer2(logits, labels)
        return


@register_task("cola-analysis", rel_path="CoLA/")
class CoLAAnalysisTask(SingleClassificationTask):
    def __init__(self, path, max_seq_len, name, **kw):
        super(CoLAAnalysisTask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]

    def load_data(self, path, max_seq_len):
        """Load the data"""
        # Load data from tsv
        tag_vocab = vocabulary.Vocabulary(counter=None)
        tr_data = load_tsv(
            tokenizer_name=self._tokenizer_name,
            data_file=os.path.join(path, "train_analysis.tsv"),
            max_seq_len=max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            tag2idx_dict={"Domain": 1},
            tag_vocab=tag_vocab,
        )
        val_data = load_tsv(
            tokenizer_name=self._tokenizer_name,
            data_file=os.path.join(path, "dev_analysis.tsv"),
            max_seq_len=max_seq_len,
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
            data_file=os.path.join(path, "test_analysis.tsv"),
            max_seq_len=max_seq_len,
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
        # Create score for each tag from tag-index dict
        self.tag_list = get_tag_list(tag_vocab)
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=Correlation, corr_type="matthews"
        )
        self.tag_scorers2 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy
        )

        log.info("\tFinished loading CoLA sperate domain.")

    def process_split(self, split, indexers):
        def _make_instance(input1, labels, tagids):
            """ from multiple types in one column create multiple fields """
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["sent1_str"] = MetadataField(" ".join(input1[1:-1]))
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            d["tagmask"] = MultiLabelField(
                tagids, label_namespace="tagids", skip_indexing=True, num_labels=len(self.tag_list)
            )
            return Instance(d)

        instances = map(_make_instance, *split)
        return instances  # lazy iterator

    def update_metrics(self, logits, labels, tagmask=None):
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
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        self.scorer2 = F1Measure(1)
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_acc_f1" % name
        self.val_metric_decreases = False

    def load_data(self, path, max_seq_len):
        """Process the dataset located at data_file."""
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=4,
            label_idx=5,
            label_fn=int,
            skip_rows=1,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=4,
            label_idx=5,
            label_fn=int,
            skip_rows=1,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=2,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
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


@register_task("mnli-fiction", rel_path="MNLI/", genre="fiction")
@register_task("mnli-slate", rel_path="MNLI/", genre="slate")
@register_task("mnli-government", rel_path="MNLI/", genre="government")
@register_task("mnli-telephone", rel_path="MNLI/", genre="telephone")
@register_task("mnli-travel", rel_path="MNLI/", genre="travel")
class MultiNLISingleGenreTask(PairClassificationTask):
    """ Task class for Multi-Genre Natural Language Inference, Fiction genre."""

    def __init__(self, path, max_seq_len, genre, name, **kw):
        """MNLI"""
        super(MultiNLISingleGenreTask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len, genre)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len, genre):
        """Process the dataset located at path. We only use the in-genre matche data."""
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=11,
            label_fn=targ_map.__getitem__,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=genre,
        )

        val_matched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev_matched.tsv"),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=11,
            label_fn=targ_map.__getitem__,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=genre,
        )

        te_matched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test_matched.tsv"),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
            filter_idx=3,
            filter_value=genre,
        )

        self.train_data_text = tr_data
        self.val_data_text = val_matched_data
        self.test_data_text = te_matched_data
        log.info("\tFinished loading MNLI " + genre + " data.")

    def get_metrics(self, reset=False):
        """ No F1 """
        return {"accuracy": self.scorer1.get_metric(reset)}


@register_task("mrpc", rel_path="MRPC/")
class MRPCTask(PairClassificationTask):
    """ Task class for Microsoft Research Paraphase Task.  """

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super(MRPCTask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        self.scorer2 = F1Measure(1)
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_acc_f1" % name
        self.val_metric_decreases = False

    def load_data(self, path, max_seq_len):
        """ Process the dataset located at path.  """
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=4,
            label_idx=0,
            skip_rows=1,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=4,
            label_idx=0,
            skip_rows=1,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=3,
            s2_idx=4,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
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


@register_task("sts-b", rel_path="STS-B/")
# second copy for different params
@register_task("sts-b-alt", rel_path="STS-B/")
class STSBTask(PairRegressionTask):
    """ Task class for Sentence Textual Similarity Benchmark.  """

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super(STSBTask, self).__init__(name, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        # self.scorer1 = Average()
        # self.scorer2 = Average()
        self.scorer1 = Correlation("pearson")
        self.scorer2 = Correlation("spearman")
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_corr" % self.name
        self.val_metric_decreases = False

    def load_data(self, path, max_seq_len):
        """ Load data """
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            skip_rows=1,
            s1_idx=7,
            s2_idx=8,
            label_idx=9,
            label_fn=lambda x: float(x) / 5,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            skip_rows=1,
            s1_idx=7,
            s2_idx=8,
            label_idx=9,
            label_fn=lambda x: float(x) / 5,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=7,
            s2_idx=8,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading STS Benchmark data.")

    def get_metrics(self, reset=False):
        pearsonr = self.scorer1.get_metric(reset)
        spearmanr = self.scorer2.get_metric(reset)
        return {"corr": (pearsonr + spearmanr) / 2, "pearsonr": pearsonr, "spearmanr": spearmanr}


@register_task("snli", rel_path="SNLI/")
class SNLITask(PairClassificationTask):
    """ Task class for Stanford Natural Language Inference """

    def __init__(self, path, max_seq_len, name, **kw):
        """ Do stuff """
        super(SNLITask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len):
        """ Process the dataset located at path.  """
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=7,
            s2_idx=8,
            label_idx=10,
            skip_rows=1,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=7,
            s2_idx=8,
            label_idx=10,
            skip_rows=1,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=7,
            s2_idx=8,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SNLI data.")


@register_task("mnli", rel_path="MNLI/")
# second copy for different params
@register_task("mnli-alt", rel_path="MNLI/")
class MultiNLITask(PairClassificationTask):
    """ Task class for Multi-Genre Natural Language Inference """

    def __init__(self, path, max_seq_len, name, **kw):
        """MNLI"""
        super(MultiNLITask, self).__init__(name, n_classes=3, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len):
        """Process the dataset located at path."""
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=11,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
        )

        # Warning to anyone who edits this: The reference label is column *15*,
        # not 11 as above.
        val_matched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev_matched.tsv"),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=15,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
        )
        val_mismatched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev_mismatched.tsv"),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            label_idx=15,
            label_fn=targ_map.__getitem__,
            skip_rows=1,
        )
        val_data = [m + mm for m, mm in zip(val_matched_data, val_mismatched_data)]
        val_data = tuple(val_data)

        te_matched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test_matched.tsv"),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        te_mismatched_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test_mismatched.tsv"),
            max_seq_len,
            s1_idx=8,
            s2_idx=9,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        te_diagnostic_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "diagnostic.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=2,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        te_data = [
            m + mm + d for m, mm, d in zip(te_matched_data, te_mismatched_data, te_diagnostic_data)
        ]

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading MNLI data.")


@register_task("mnli-diagnostic", rel_path="MNLI/")
class MultiNLIDiagnosticTask(PairClassificationTask):
    """ Task class for diagnostic on MNLI"""

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=3, **kw)
        self.load_data_and_create_scorers(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data_and_create_scorers(self, path, max_seq_len):
        """load MNLI diagnostics data. The tags for every column are loaded as indices.
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

        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        diag_data_dic = load_diagnostic_tsv(
            self._tokenizer_name,
            os.path.join(path, "diagnostic-full.tsv"),
            max_seq_len,
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
        log.info("\tFinished loading MNLI Diagnostics data.")

        # TODO: use FastMatthews instead to save memory.
        create_score_function(Correlation, "matthews", self.ix_to_lex_sem_dic, "lex_sem")
        create_score_function(Correlation, "matthews", self.ix_to_pr_ar_str_dic, "pr_ar_str")
        create_score_function(Correlation, "matthews", self.ix_to_logic_dic, "logic")
        create_score_function(Correlation, "matthews", self.ix_to_knowledge_dic, "knowledge")
        log.info("\tFinished creating Score functions for Diagnostics data.")

    def update_diagnostic_metrics(self, logits, labels, batch):
        # Updates scorer for every tag in a given column (tag_group) and also the
        # the scorer for the column itself.
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
                sub_logits = logits[indices_to_pull[:, 0]]
                scorer = getattr(self, scorer_str)
                scorer(sub_logits, sub_labels)
            return

        # Updates scorers for each tag.
        update_scores_for_tag_group(self.ix_to_lex_sem_dic, "lex_sem")
        update_scores_for_tag_group(self.ix_to_pr_ar_str_dic, "pr_ar_str")
        update_scores_for_tag_group(self.ix_to_logic_dic, "logic")
        update_scores_for_tag_group(self.ix_to_knowledge_dic, "knowledge")

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        is_using_bert = "bert_wpm_pretokenized" in indexers

        def create_labels_from_tags(fields_dict, ix_to_tag_dict, tag_arr, tag_group):
            # If there is something in this row then tag_group should be set to
            # 1.
            is_tag_group = 1 if len(tag_arr) != 0 else 0
            fields_dict[tag_group] = LabelField(
                is_tag_group, label_namespace=tag_group, skip_indexing=True
            )
            # For every possible tag in the column set 1 if the tag is present for
            # this example, 0 otherwise.
            for ix, tag in ix_to_tag_dict.items():
                if ix == 0:
                    continue
                is_present = 1 if ix in tag_arr else 0
                fields_dict["%s__%s" % (tag_group, tag)] = LabelField(
                    is_present, label_namespace="%s__%s" % (tag_group, tag), skip_indexing=True
                )
            return

        def _make_instance(input1, input2, label, idx, lex_sem, pr_ar_str, logic, knowledge):
            """ from multiple types in one column create multiple fields """
            d = {}
            if is_using_bert:
                inp = input1 + input2[1:]  # drop the leading [CLS] token
                d["inputs"] = sentence_to_text_field(inp, indexers)
            else:
                d["input1"] = sentence_to_text_field(input1, indexers)
                d["input2"] = sentence_to_text_field(input2, indexers)
            d["labels"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idx", skip_indexing=True)
            d["sent1_str"] = MetadataField(" ".join(input1[1:-1]))
            d["sent2_str"] = MetadataField(" ".join(input2[1:-1]))

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
        # We do not compute accuracy for this dataset but the eval function
        # requires this key.
        collected_metrics["accuracy"] = 0

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
        return collected_metrics


@register_task("rte", rel_path="RTE/")
class RTETask(PairClassificationTask):
    """ Task class for Recognizing Textual Entailment 1, 2, 3, 5 """

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super().__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len):
        """ Process the datasets located at path. """
        targ_map = {"not_entailment": 0, "entailment": 1}
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=2,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading RTE (from GLUE formatted data).")


@register_task('rte-superglue', rel_path='RTE/')
class RTESuperGLUETask(RTETask):
    ''' Task class for Recognizing Textual Entailment 1, 2, 3, 5 '''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' '''
        super().__init__(path, max_seq_len, name, **kw)

    def load_data(self, path, max_seq_len):
        ''' Process the datasets located at path. '''
        targ_map = {"not_entailment": 0, "entailment": 1}
        def _load_jsonl(data_file):
            data = [json.loads(d) for d in open(data_file, encoding="utf-8")]
            sent1s, sent2s, trgs, idxs = [], [], [], []
            for example in data:
                sent1s.append(process_sentence(self._tokenizer_name, example["premise"], max_seq_len))
                sent2s.append(process_sentence(self._tokenizer_name, example["hypothesis"], max_seq_len))
                trg = targ_map[example["label"]] if "label" in example else 0
                trgs.append(trg)
                idxs.append(example["idx"])
            return [sent1s, sent2s, trgs, idxs]

        tr_data = _load_jsonl(os.path.join(path, "train.jsonl"))
        val_data = _load_jsonl(os.path.join(path, "val.jsonl"))
        te_data = _load_jsonl(os.path.join(path, "test_ANS.jsonl"))
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading RTE (from SuperGLUE formatted data).")


@register_task("qnli", rel_path="QNLI/")
# second copy for different params
@register_task("qnli-alt", rel_path="QNLI/")
class QNLITask(PairClassificationTask):
    """Task class for SQuAD NLI"""

    def __init__(self, path, max_seq_len, name, **kw):
        super(QNLITask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len):
        """Load the data"""
        targ_map = {"not_entailment": 0, "entailment": 1}
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            label_fn=targ_map.__getitem__,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=2,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading QNLI.")


@register_task("wnli", rel_path="WNLI/")
class WNLITask(PairClassificationTask):
    """Class for Winograd NLI task"""

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super(WNLITask, self).__init__(name, n_classes=2, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len):
        """Load the data"""
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            skip_rows=1,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=2,
            has_labels=False,
            return_indices=True,
            skip_rows=1,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading Winograd.")


@register_task("joci", rel_path="JOCI/")
class JOCITask(PairOrdinalRegressionTask):
    """Class for JOCI ordinal regression task"""

    def __init__(self, path, max_seq_len, name, **kw):
        super(JOCITask, self).__init__(name, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len):
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            skip_rows=1,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            skip_rows=1,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            skip_rows=1,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
        )
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading JOCI data.")


@register_task("wiki103_classif", rel_path="WikiText103/")
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

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.
        Split should be one of 'train', 'val', or 'test'.
        """
        return self.load_data(self.files_by_split[split])

    def load_data(self, path):
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
            for sent in self.load_data(path):
                yield sent

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process a language modeling split.  Split is a single list of sentences here.  """

        def _make_instance(input1, input2, labels):
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["input2"] = sentence_to_text_field(input2, indexers)
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

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.

        Split should be one of 'train', 'val', or 'test'.
        """
        return self.load_data(self.files_by_split[split])

    def load_data(self, path):
        """ Load data """
        with open(path, "r") as txt_fh:
            for row in txt_fh:
                row = row.strip().split("\t")
                if len(row) != 3 or not (row[0] and row[1] and row[2]):
                    continue
                sent1 = process_sentence(self._tokenizer_name, row[0], self.max_seq_len)
                sent2 = process_sentence(self._tokenizer_name, row[1], self.max_seq_len)
                targ = int(row[2])
                yield (sent1, sent2, targ)

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            """ Don't use test set for vocab building. """
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for sent1, sent2, _ in self.load_data(path):
                yield sent1
                yield sent2

    def count_examples(self):
        """ Compute the counts here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(1 for line in open(split_path))
        self.example_counts = example_counts

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        is_using_bert = "bert_wpm_pretokenized" in indexers

        def _make_instance(input1, input2, labels):
            d = {}
            if is_using_bert:
                inp = input1 + input2[1:]  # drop leading [CLS] token
                d["inputs"] = sentence_to_text_field(inp, indexers)
            else:
                d["input1"] = sentence_to_text_field(input1, indexers)
                d["input2"] = sentence_to_text_field(input2, indexers)
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            return Instance(d)

        for sent1, sent2, trg in split:
            yield _make_instance(sent1, sent2, trg)


# TODO: does this even work? What is n_classes for this?
@register_task("weakgrounded", rel_path="mscoco/weakgrounded/")
class WeakGroundedTask(PairClassificationTask):
    """ Task class for Weak Grounded Sentences i.e., training on pairs of captions for the same image """

    def __init__(self, path, max_seq_len, n_classes, name, **kw):
        """ Do stuff """
        super(WeakGroundedTask, self).__init__(name, n_classes, **kw)

        """ Process the dataset located at path.  """
        """ positive = captions of the same image, negative = captions of different images """
        targ_map = {"negative": 0, "positive": 1}
        targ_map = {"0": 0, "1": 1}

        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train_aug.tsv"),
            max_seq_len,
            targ_map=targ_map,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
            skip_rows=0,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "val.tsv"),
            max_seq_len,
            targ_map=targ_map,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
            skip_rows=0,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            targ_map=targ_map,
            s1_idx=0,
            s2_idx=1,
            label_idx=2,
            skip_rows=0,
        )

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.n_classes = 2
        log.info("\tFinished loading MSCOCO data.")


@register_task("grounded", rel_path="mscoco/grounded/")
class GroundedTask(Task):
    """ Task class for Grounded Sentences i.e., training on caption->image pair """

    """ Defined new metric function from AllenNLP Average """

    def __init__(self, path, max_seq_len, name, **kw):
        """ Do stuff """
        super(GroundedTask, self).__init__(name, **kw)
        self.scorer1 = Average()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_metric" % self.name
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.ids = self.train_data_text[1] + self.val_data_text[1]
        self.path = path
        self.img_encoder = None
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        metric = self.scorer1.get_metric(reset)

        return {"metric": metric}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """
        Convert a dataset of sentences into padded sequences of indices.
        Args:
            - split (list[list[str]]): list of inputs (possibly pair) and outputs
            - pair_input (int)
            - tok2idx (dict)
        Returns:
        """

        def _make_instance(sent, label, ids):
            input1 = sentence_to_text_field(sent, indexers)
            label = NumericField(label)
            ids = NumericField(ids)
            return Instance({"input1": input1, "labels": label, "ids": ids})

        # Map over columns: input1, labels, ids
        instances = map(_make_instance, *split)
        #  return list(instances)
        return instances  # lazy iterator

    def load_data(self, path, max_seq_len):
        """ Map sentences to image ids
            Keep track of caption ids just in case """

        train, val, test = ([], [], []), ([], [], []), ([], [], [])

        with open(os.path.join(path, "train_idx.txt"), "r") as f:
            train_ids = [item.strip() for item in f.readlines()]
        with open(os.path.join(path, "val_idx.txt"), "r") as f:
            val_ids = [item.strip() for item in f.readlines()]
        with open(os.path.join(path, "test_idx.txt"), "r") as f:
            test_ids = [item.strip() for item in f.readlines()]

        f = open(os.path.join(path, "train.json"), "r")
        for line in f:
            tr_dict = json.loads(line)
        f = open(os.path.join(path, "val.json"), "r")
        for line in f:
            val_dict = json.loads(line)
        f = open(os.path.join(path, "test.json"), "r")
        for line in f:
            te_dict = json.loads(line)
        with open(os.path.join(path, "feat_map.json")) as fd:
            keymap = json.load(fd)

        def load_mscoco(data_dict, data_list, img_idxs):
            for img_idx in img_idxs:
                newimg_id = "mscoco/grounded/" + img_idx + ".json"
                for caption_id in data_dict[img_idx]["captions"]:
                    data_list[0].append(data_dict[img_idx]["captions"][caption_id])
                    data_list[1].append(1)
                    data_list[2].append(int(keymap[newimg_id]))
            return data_list

        train = load_mscoco(tr_dict, train, train_ids)
        val = load_mscoco(val_dict, val, val_ids)
        test = load_mscoco(te_dict, test, test_ids)

        self.tr_data = train
        self.val_data = val
        self.te_data = test
        self.train_data_text = train
        self.val_data_text = val
        self.test_data_text = test

        log.info("\tTrain: %d, Val: %d, Test: %d", len(train[0]), len(val[0]), len(test[0]))
        log.info("\tFinished loading MSCOCO data!")


@register_task("groundedsw", rel_path="mscoco/grounded")
class GroundedSWTask(Task):
    """ Task class for Grounded Sentences i.e., training on caption->image pair """

    """ Defined new metric function from AllenNLP Average """

    def __init__(self, path, max_seq_len, name, **kw):
        super(GroundedSWTask, self).__init__(name, **kw)
        self.scorer1 = Average()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_metric" % self.name
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.ids = self.train_data_text[1] + self.val_data_text[1]
        self.path = path
        self.img_encoder = None
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        metric = self.scorer1.get_metric(reset)

        return {"metric": metric}

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """
        Convert a dataset of sentences into padded sequences of indices.
        Args:
            - split (list[list[str]]): list of inputs (possibly pair) and outputs
            - pair_input (int)
            - tok2idx (dict)
        Returns:
        """

        def _make_instance(sent, label, ids):
            input1 = sentence_to_text_field(sent, indexers)
            label = NumericField(label)
            ids = NumericField(ids)
            return Instance({"input1": input1, "labels": label, "ids": ids})

        # Map over columns: input1, labels, ids
        instances = map(_make_instance, *split)
        #  return list(instances)
        return instances  # lazy iterator

    def load_data(self, path, max_seq_len):
        """ Map sentences to image ids
            Keep track of caption ids just in case """

        train, val, test = ([], [], []), ([], [], []), ([], [], [])

        def get_data(dataset, data):
            f = open(path + dataset + ".tsv", "r")
            for line in f:
                items = line.strip().split("\t")
                if len(items) < 3 or items[1] == "0":
                    continue
                data[0].append(items[0])
                data[1].append(int(items[1]))
                data[2].append(int(items[2]))
            return data

        train = get_data("shapeworld/train", train)
        val = get_data("shapeworld/val", val)
        test = get_data("shapeworld/test", test)

        self.tr_data = train
        self.val_data = val
        self.te_data = test
        self.train_data_text = train
        self.val_data_text = val
        self.test_data_text = test

        log.info("Train: %d, Val: %d, Test: %d", len(train[0]), len(val[0]), len(test[0]))
        log.info("\nFinished loading SW data!")


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
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len):
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "train.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=2,
            skip_rows=0,
            label_idx=3,
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "dev.tsv"),
            max_seq_len,
            s1_idx=0,
            s2_idx=1,
            skip_rows=0,
            label_idx=3,
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "test.tsv"),
            max_seq_len,
            s1_idx=1,
            s2_idx=2,
            skip_rows=0,
            label_idx=3,
        )

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading recast probing data.")


class TaggingTask(Task):
    """ Generic tagging task, one tag per word """

    def __init__(self, name, num_tags, **kw):
        super().__init__(name, **kw)
        assert num_tags > 0
        self.num_tags = num_tags + 2  # add tags for unknown and padding
        self.scorer1 = CategoricalAccuracy()
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.all_labels = [str(i) for i in range(self.num_tags)]
        self._label_namespace = self.name + "_tags"
        self.target_indexer = {"words": SingleIdTokenIndexer(namespace=self._label_namespace)}

    def truncate(self, max_seq_len, sos_tok=utils.SOS_TOK, eos_tok=utils.EOS_TOK):
        """ Truncate the data if any sentences are longer than max_seq_len. """
        self.train_data_text = [
            truncate(self.train_data_text[0], max_seq_len, sos_tok, eos_tok),
            self.train_data_text[1],
        ]
        self.val_data_text = [
            truncate(self.val_data_text[0], max_seq_len, sos_tok, eos_tok),
            self.val_data_text[1],
        ]
        self.test_data_text = [
            truncate(self.test_data_text[0], max_seq_len, sos_tok, eos_tok),
            self.test_data_text[1],
        ]

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}

    def get_all_labels(self) -> List[str]:
        return self.all_labels


@register_task("ccg", rel_path="CCG/")
class CCGTaggingTask(TaggingTask):
    """ CCG supertagging as a task.
        Using the supertags from CCGbank. """

    def __init__(self, path, max_seq_len, name="ccg", **kw):
        """ There are 1363 supertags in CCGBank without introduced token. """
        super().__init__(name, 1363, **kw)
        self.INTRODUCED_TOKEN = "1363"
        self.bert_tokenization = self._tokenizer_name.startswith("bert-")
        self.load_data(path, max_seq_len)
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        self.max_seq_len = max_seq_len
        if self._tokenizer_name.startswith("bert-"):
            # the +1 is for the tokenization added token
            self.num_tags = self.num_tags + 1

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process a tagging task """
        inputs = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        targs = [
            TextField(list(map(Token, sent)), token_indexers=self.target_indexer)
            for sent in split[2]
        ]
        mask = [
            MultiLabelField(mask, label_namespace="indices", skip_indexing=True, num_labels=511)
            for mask in split[3]
        ]
        instances = [
            Instance({"inputs": x, "targs": t, "mask": m}) for (x, t, m) in zip(inputs, targs, mask)
        ]
        return instances

    def load_data(self, path, max_seq_len):
        tr_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "ccg.train." + self._tokenizer_name),
            max_seq_len,
            s1_idx=1,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            col_indices=[0, 1, 2],
            delimiter="\t",
            label_fn=lambda t: t.split(" "),
        )
        val_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "ccg.dev." + self._tokenizer_name),
            max_seq_len,
            s1_idx=1,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            col_indices=[0, 1, 2],
            delimiter="\t",
            label_fn=lambda t: t.split(" "),
        )
        te_data = load_tsv(
            self._tokenizer_name,
            os.path.join(path, "ccg.test." + self._tokenizer_name),
            max_seq_len,
            s1_idx=1,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            col_indices=[0, 1, 2],
            delimiter="\t",
            has_labels=False,
        )
        self.max_seq_len = max_seq_len

        # Get the mask for each sentence, where the mask is whether or not
        # the token was split off by tokenization. We want to only count the first
        # sub-piece in the BERT tokenization in the loss and score, following Devlin's NER
        # experiment [BERT: Pretraining of Deep Bidirectional Transformers for Language Understanding]
        # (https://arxiv.org/abs/1810.04805)
        if self.bert_tokenization:
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
        log.info("\tFinished loading CCGTagging data.")


class SpanClassificationTask(Task):
    """
     Generic class for span tasks.
    Acts as a classifier, but with multiple targets for each input text.
    Targets are of the form (span1, span2,..., span_n, label), where the spans are
    half-open token intervals [i, j).
    The number of spans is constant across examples.
    """

    @property
    def _tokenizer_suffix(self):
        """"
        Suffix to make sure we use the correct source files,
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
            split: os.path.join(path, fname) + self._tokenizer_suffix
            for split, fname in files_by_split.items()
        }
        self.num_spans = num_spans
        self._iters_by_split = self.load_data()
        self.max_seq_len = max_seq_len

        label_file = os.path.join(path, label_file)
        self.all_labels = list(utils.load_lines(label_file))
        self.n_classes = len(self.all_labels)
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

    def load_data(self):
        iters_by_split = collections.OrderedDict()
        for split, filename in self._files_by_split.items():
            iter = list(self._stream_records(filename))
            iters_by_split[split] = iter
        return iters_by_split

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

    def _pad_tokens(self, tokens):
        """Pad tokens according to the current tokenization style."""
        if self.tokenizer_name.startswith("bert-"):
            # standard padding for BERT; see
            # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/extract_features.py#L85
            return ["[CLS]"] + tokens + ["[SEP]"]
        else:
            return [utils.SOS_TOK] + tokens + [utils.EOS_TOK]

    def make_instance(self, record, idx, indexers) -> Type[Instance]:
        """Convert a single record to an AllenNLP Instance."""
        tokens = record["text"].split()
        tokens = self._pad_tokens(tokens)
        text_field = sentence_to_text_field(tokens, indexers)

        example = {}
        example["idx"] = MetadataField(idx)

        example["input1"] = text_field

        for i in range(self.num_spans):
            example["span" + str(i + 1) + "s"] = ListField(
                [
                    self._make_span_field(t["span" + str(i + 1)], text_field, 1)
                    for t in record["targets"]
                ]
            )

        labels = [utils.wrap_singleton_string(t["label"]) for t in record["targets"]]
        example["labels"] = ListField(
            [
                MultiLabelField(
                    label_set, label_namespace=self._label_namespace, skip_indexing=False
                )
                for label_set in labels
            ]
        )
        return Instance(example)

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
        metrics["acc"] = self.acc_scorer.get_metric(reset)
        precision, recall, f1 = self.f1_scorer.get_metric(reset)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        return metrics


@register_task("commitbank", rel_path="CommitmentBank/")
class CommitmentTask(PairClassificationTask):
    """ NLI-formatted task detecting speaker commitment.
    Data and more info at github.com/mcdm/CommitmentBank/
    Paper forthcoming. """

    def __init__(self, path, max_seq_len, name, **kw):
        """ We use three F1 trackers, one for each class to compute multi-class F1 """
        super().__init__(name, n_classes=3, **kw)
        self.scorer2 = F1Measure(0)
        self.scorer3 = F1Measure(1)
        self.scorer4 = F1Measure(2)
        self.scorers = [self.scorer1, self.scorer2, self.scorer3, self.scorer4]
        self.val_metric = "%s_f1" % name

        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[1]
        )

    def load_data(self, path, max_seq_len):
        """Process the dataset located at each data file.
           The target needs to be split into tokens because
           it is a sequence (one tag per input token). """
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}

        def _load_data(data_file):
            data = [json.loads(l) for l in open(data_file, encoding="utf-8").readlines()]
            sent1s, sent2s, targs, idxs = [], [], [], []
            for example in data:
                sent1s.append(
                    process_sentence(self._tokenizer_name, example["premise"], max_seq_len)
                )
                sent2s.append(
                    process_sentence(self._tokenizer_name, example["hypothesis"], max_seq_len)
                )
                trg = targ_map[example["label"]] if "label" in example else 0
                targs.append(trg)
                targs.append(trg)
                idxs.append(example["idx"])
            return [sent1s, sent2s, targs, idxs]

        tr_data = _load_data(os.path.join(path, "train.jsonl"))
        val_data = _load_data(os.path.join(path, "val.jsonl"))
        te_data = _load_data(os.path.join(path, "test_ANS.jsonl"))

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
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
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.train_data_text[1]
            + self.val_data_text[0]
            + self.val_data_text[1]
        )
        self.scorer1 = CategoricalAccuracy()
        self.scorer2 = F1Measure(1)
        self.scorers = [self.scorer1, self.scorer2]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False

    def load_data(self, path, max_seq_len):
        """Process the dataset located at data_file."""

        trg_map = {"true": 1, "false": 0, True: 1, False: 0}

        def _load_split(data_file):
            sents1, sents2, idxs1, idxs2, trgs = [], [], [], [], []
            with open(data_file, "r") as data_fh:
                for row in data_fh:
                    row = json.loads(row)
                    sent1 = process_sentence(self._tokenizer_name, row["sentence1"], max_seq_len)
                    sent2 = process_sentence(self._tokenizer_name, row["sentence2"], max_seq_len)
                    sents1.append(sent1)
                    sents2.append(sent2)
                    idx1 = row["sentence1_idx"]
                    idx2 = row["sentence2_idx"]
                    idxs1.append(int(idx1))
                    idxs2.append(int(idx2))
                    trg = trg_map[row["label"]] if "label" in row else 0
                    trgs.append(trg)
                return [sents1, sents2, idxs1, idxs2, trgs]

        tr_data = _load_split(os.path.join(path, "train.jsonl"))
        val_data = _load_split(os.path.join(path, "val.jsonl"))
        te_data = _load_split(os.path.join(path, "test.jsonl"))
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading WiC data.")

    def process_split(self, split, indexers):
        """
        Convert a dataset of sentences into padded sequences of indices. Shared
        across several classes.

        """
        # check here if using bert to avoid passing model info to tasks
        is_using_bert = "bert_wpm_pretokenized" in indexers

        def _make_instance(input1, input2, idxs1, idxs2, labels, idx):
            d = {}
            d["sent1_str"] = MetadataField(" ".join(input1[1:-1]))
            d["idx1"] = NumericField(idxs1)
            d["sent2_str"] = MetadataField(" ".join(input2[1:-1]))
            d["idx2"] = NumericField(idxs2)  # modify if using BERT
            if is_using_bert:
                inp = input1 + input2[1:]  # throw away input2 leading [CLS]
                d["inputs"] = sentence_to_text_field(inp, indexers)
                idxs2 += len(input1)
            else:
                d["input1"] = sentence_to_text_field(input1, indexers)
                d["input2"] = sentence_to_text_field(input2, indexers)
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)

            d["idx"] = LabelField(idx, label_namespace="idxs", skip_indexing=True)

            return Instance(d)

        if len(split) < 6:  # counting iterator for idx
            assert len(split) == 5
            split.append(itertools.count())

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

    pass


@register_task("copa", rel_path="COPA/")
class COPATask(MultipleChoiceTask):
    """ Task class for Choice of Plausible Alternatives Task.  """

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
        super().__init__(name, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + [choice for choices in self.train_data_text[1] for choice in choices]
            + [choice for choices in self.val_data_text[1] for choice in choices]
        )
        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False
        self.n_choices = 2

    def load_data(self, path, max_seq_len):
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
                    process_sentence(self._tokenizer_name, choice, max_seq_len)
                    for choice in [choice1, choice2]
                ]
                targ = example["label"] if "label" in example else 0
                contexts.append(process_sentence(self._tokenizer_name, context, max_seq_len))
                choicess.append(choices)
                questions.append(process_sentence(self._tokenizer_name, question, max_seq_len))
                targs.append(targ)
            return [contexts, choicess, questions, targs]

        self.train_data_text = _load_split(os.path.join(path, "train.jsonl"))
        self.val_data_text = _load_split(os.path.join(path, "val.jsonl"))
        self.test_data_text = _load_split(os.path.join(path, "test_ANS.jsonl"))
        log.info("\tFinished loading COPA (as QA) data.")

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AlleNNLP Instances. """
        is_using_bert = "bert_wpm_pretokenized" in indexers

        def _make_instance(context, choices, question, label, idx):
            d = {}
            d["question_str"] = MetadataField(" ".join(context[1:-1]))
            if not is_using_bert:
                d["question"] = sentence_to_text_field(context, indexers)
            for choice_idx, choice in enumerate(choices):
                inp = context + question[1:] + choice[1:] if is_using_bert else choice
                d["choice%d" % choice_idx] = sentence_to_text_field(inp, indexers)
                d["choice%d_str" % choice_idx] = MetadataField(" ".join(choice[1:-1]))
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idxs", skip_indexing=True)
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
        """ """
        super().__init__(name, **kw)
        self.load_data(path, max_seq_len)
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + [choice for choices in self.train_data_text[1] for choice in choices]
            + [choice for choices in self.val_data_text[1] for choice in choices]
        )
        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False
        self.n_choices = 4

    def load_data(self, path, max_seq_len):
        """ Process the dataset located at path.  """

        def _load_split(data_file):
            questions, choicess, targs = [], [], []
            data = pd.read_csv(data_file)
            for ex_idx, ex in data.iterrows():
                sent1 = process_sentence(self._tokenizer_name, ex["sent1"], max_seq_len)
                questions.append(sent1)
                sent2_prefix = ex["sent2"]
                choices = []
                for i in range(4):
                    choice = sent2_prefix + " " + ex["ending%d" % i]
                    choice = process_sentence(self._tokenizer_name, choice, max_seq_len)
                    choices.append(choice)
                choicess.append(choices)
                targ = ex["label"] if "label" in ex else 0
                targs.append(targ)
            return [questions, choicess, targs]

        self.train_data_text = _load_split(os.path.join(path, "train.csv"))
        self.val_data_text = _load_split(os.path.join(path, "val.csv"))
        self.test_data_text = _load_split(os.path.join(path, "test.csv"))
        log.info("\tFinished loading SWAG data.")

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AlleNNLP Instances. """
        is_using_bert = "bert_wpm_pretokenized" in indexers

        def _make_instance(question, choices, label, idx):
            d = {}
            d["question_str"] = MetadataField(" ".join(question[1:-1]))
            if not is_using_bert:
                d["question"] = sentence_to_text_field(question, indexers)
            for choice_idx, choice in enumerate(choices):
                inp = question + choice[1:] if is_using_bert else choice
                d["choice%d" % choice_idx] = sentence_to_text_field(inp, indexers)
                d["choice%d_str" % choice_idx] = MetadataField(" ".join(choice[1:-1]))
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["idx"] = LabelField(idx, label_namespace="idxs", skip_indexing=True)
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
