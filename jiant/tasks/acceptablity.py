"""
Task definition for acceptability probing tasks
"""
import collections
import copy
import json
import logging as log
import os

import numpy as np
import torch
import IPython


# Fields for instance processing
from allennlp.data import Instance, Token, vocabulary
from allennlp.data.fields import (
    LabelField,
    MetadataField,
    MultiLabelField,
    IndexField,
    TextField
)
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.training.metrics import Average, CategoricalAccuracy

from jiant.allennlp_mods.correlation import Correlation
from jiant.allennlp_mods.numeric_field import NumericField
from jiant.utils import utils
from jiant.utils.data_loaders import (
    load_span_data,
    load_tsv,
    tokenize_and_truncate,
    TagManager
)
from jiant.utils.tokenizers import get_tokenizer
from jiant.tasks.registry import register_task  # global task registry
from jiant.metrics.winogender_metrics import GenderParity
from jiant.tasks.tasks import (
    Task,
    SingleClassificationTask,
    PairClassificationTask,
    sentence_to_text_field,
    create_subset_scorers,
    update_subset_scorers,
    collect_subset_scores
)


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
@register_task("wilcox_npi", rel_path="NPI/wilcox")
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
    """Class for NPI-related task; same with Warstdadt acceptability task but outputs labels for
       test-set
       Note: Used for an NYU seminar, data not yet public"""

    def __init__(self, path, max_seq_len, name, **kw):
        """ """
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
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]

        self.tag_manager = TagManager()
        self.tag_list = None
        self.tag_scorers1 = None
        self.tag_scorers2 = None

    def load_data(self):
        """Load the data"""
        # Load data from tsv
        tr_data = load_tsv(
            tokenizer_name=self._tokenizer_name,
            data_file=os.path.join(self.path, "train_analysis.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=3,
            s2_idx=None,
            label_idx=2,
            skip_rows=1,
            tag2idx_dict={"Domain": 1},
            tag_manager=self.tag_manager,
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
            tag_manager=self.tag_manager,
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
            tag_manager=self.tag_manager,
        )
        self.train_data_text = tr_data[:1] + tr_data[2:]
        self.val_data_text = val_data[:1] + val_data[2:]
        self.test_data_text = te_data[:1] + te_data[2:]
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        # Create score for each tag from tag-index dict
        self.tag_list = self.tag_managerget_tag_list()
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=Correlation, corr_type="matthews"
        )
        self.tag_scorers2 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy
        )
        log.info("\tFinished loading CoLA sperate domain.")

    def process_split(self, split, indexers, boundary_token_fn):
        def _make_instance(input1, labels, tagids):
            """ from multiple types in one column create multiple fields """
            d = {}
            d["input1"] = sentence_to_text_field(boundary_token_fn(input1), indexers)
            d["sent1_str"] = MetadataField(" ".join(input1))
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            d["tagmask"] = MultiLabelField(
                tagids, label_namespace="tags", skip_indexing=True, num_labels=len(self.tag_list)
            )
            return Instance(d)

        instances = map(_make_instance, *split)
        return instances

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


class BlimpTask(PairClassificationTask):
    """ Class for linguistic phenomena pair tasks """

    def __init__(self, path, max_seq_len, name, **kw):
        super(BlimpTask, self).__init__(name, n_classes=2, **kw)        
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None
        self.eval_only_task = True
        
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        
        self.tag_manager = TagManager()
        self.tag_list = None
        self.tag_scorers1 = None

    def update_metrics(self, logits, labels, tagmask=None):
        logits, labels = logits.detach(), labels.detach()
        self.scorer1(logits, labels)
        if tagmask is not None:
            update_subset_scorers(self.tag_scorers1, logits, labels, tagmask)
        return

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""

        collected_metrics = {
            "accuracy": self.scorer1.get_metric(reset),
        }
        collected_metrics.update(
            collect_subset_scores(self.tag_scorers1, "accuracy", self.tag_list, reset)
        )
        return collected_metrics


@register_task("blimp-oneprefix", rel_path="blimp")
class BlimpOnePrefixLMTask(BlimpTask):
    """ Task class for full sentence LM acceptability preference """

    def __init__(self, path, max_seq_len, name, **kw):
        super(BlimpOnePrefixLMTask, self).__init__(path, max_seq_len, name, **kw)

    def load_data(self):
        """ Load linguistic phenomena benchmark data for one prefix method, each one prefix
        example includes one shared prefix and the following tokens from the good and bad
        sentences """
        data_file = os.path.join(self.path, "blimp.jsonl")
        data = [json.loads(l) for l in open(data_file, encoding="utf-8").readlines()]
        tag_types = ['field', 'linguistics_term', 'UID']
        sent1s, sent2s, labels, tags = [], [], [], []
        shared_prefixes, good_words, bad_words = [], [], []
        for example in data:
            if not example['one_prefix_method']:
                continue
            sent1s.append(
                tokenize_and_truncate(self._tokenizer_name, example["sentence_good"], self.max_seq_len)
            )
            sent2s.append(
                tokenize_and_truncate(self._tokenizer_name, example["sentence_bad"], self.max_seq_len)
            )
            labels.append(0)
            tags.append([])
            for tag_type in tag_types:
                if tag_type in example:
                    tag_str = "%s__%s" % (tag_type, example[tag_type])
                    tags[-1].append(self.tag_manager(tag_str))
            shared_prefixes.append(
                tokenize_and_truncate(self._tokenizer_name, example["one_prefix_prefix"], self.max_seq_len)
            )
            good_words.append(
                 tokenize_and_truncate(self._tokenizer_name, example["one_prefix_word_good"], self.max_seq_len)
            )
            bad_words.append(
                 tokenize_and_truncate(self._tokenizer_name, example["one_prefix_word_bad"], self.max_seq_len)
            )
        self.val_data_text = self.test_data_text = self.train_data_text = \
            (sent1s, sent2s, labels, tags, shared_prefixes, good_words, bad_words)
        self.sentences = self.train_data_text[0] + self.train_data_text[1]
        
        self.tag_list = self.tag_manager.get_tag_list()
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy
        )

        log.info("\tFinished loading blimp one prefix data.")
        return

    def process_split(self, split, indexers, boundary_token_fn):
        def _make_instance(sent1, sent2, label, tags, shared_prefix, good_word, bad_word):
            """ from multiple types in one column create multiple fields
            sent1: sentence1, the good one
            sent2: sentence2, the bad one
            label: always 0
            shared_prefix: shared part of both sentence
            good_word: tokens of the following word in good sentence
            bad_word: tokens of the following word in bad sentence
            tagmask: which tags this sample has
            """
            d = {}
            input1 = boundary_token_fn(sent1)
            input2 = boundary_token_fn(sent2)
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["sent1_str"] = MetadataField(" ".join(sent1))
            d["input2"] = sentence_to_text_field(input2, indexers)
            d["sent2_str"] = MetadataField(" ".join(sent2))
            d["labels"] = LabelField(label, label_namespace="label", skip_indexing=True)
            d["shared_prefix_length"] = IndexField(
                common_prefix_length(boundary_token_fn(shared_prefix), input1), d["input1"]
            )
            d["shared_str"] = MetadataField(" ".join(shared_prefix))
            d["good_word_length"] = IndexField(len(good_word), d["input1"])
            d["bad_word_length"] = IndexField(len(bad_word), d["input2"])
            d["tagmask"] = MultiLabelField(
                tags, label_namespace="tags", skip_indexing=True, num_labels=len(self.tag_list)
            )
            return Instance(d)

        instances = map(_make_instance, *split)
        return instances


@register_task("blimp-twoprefix", rel_path="blimp")
class BlimpTwoPrefixLMTask(BlimpTask):
    """ Task class for two prefix LM acceptability preference """

    def __init__(self, path, max_seq_len, name, **kw):
        super(BlimpTwoPrefixLMTask, self).__init__(path, max_seq_len, name, **kw)

    def load_data(self):
        """ Load linguistic phenomena benchmark data for two prefix method, each two prefix
        example includes one shared token and the prefix tokens from the good and bad
        sentences """
        data_file = os.path.join(self.path, "blimp.jsonl")
        data = [json.loads(l) for l in open(data_file, encoding="utf-8").readlines()]
        tag_types = ['field', 'linguistics_term', 'UID']
        sent1s, sent2s, labels, tags = [], [], [], []
        good_prefixes, bad_prefixes, shared_words = [], [], []
        for example in data:
            if not example['two_prefix_method']:
                continue
            sent1s.append(
                tokenize_and_truncate(self._tokenizer_name, example["sentence_good"], self.max_seq_len)
            )
            sent2s.append(
                tokenize_and_truncate(self._tokenizer_name, example["sentence_bad"], self.max_seq_len)
            )
            labels.append(0)
            tags.append([])
            for tag_type in tag_types:
                if tag_type in example:
                    tag_str = "%s__%s" % (tag_type, example[tag_type])
                    tags[-1].append(self.tag_manager(tag_str))
            good_prefixes.append(
                tokenize_and_truncate(self._tokenizer_name, example["two_prefix_prefix_good"], self.max_seq_len)
            )
            bad_prefixes.append(
                tokenize_and_truncate(self._tokenizer_name, example["two_prefix_prefix_bad"], self.max_seq_len)
            )
            shared_words.append(
                tokenize_and_truncate(self._tokenizer_name, example["two_prefix_word"], self.max_seq_len)
            )
        self.val_data_text = self.test_data_text = self.train_data_text = \
            (sent1s, sent2s, labels, tags, good_prefixes, bad_prefixes, shared_words)
        self.sentences = self.train_data_text[0] + self.train_data_text[1]
        
        self.tag_list = self.tag_manager.get_tag_list()
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy
        )

        log.info("\tFinished loading blimp two prefix data.")
        return

    def process_split(self, split, indexers, boundary_token_fn):
        def _make_instance(sent1, sent2, label, tags, good_prefix, bad_prefix, shared_word):
            """ from multiple types in one column create multiple fields
            sent1: sentence1, the good one
            sent2: sentence2, the bad one
            label: always 0
            good_prefix_length: length of prefix of the good sentence
            bad_prefix_length: length of prefix of the bad sentence
            shared_word_length: number of tokens forming the shared word following the prefixes
            tagmask: which tags this sample has
            """
            d = {}
            input1 = boundary_token_fn(sent1)
            input2 = boundary_token_fn(sent2)
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["sent1_str"] = MetadataField(" ".join(sent1))
            d["input2"] = sentence_to_text_field(input2, indexers)
            d["sent2_str"] = MetadataField(" ".join(sent2))
            d["labels"] = LabelField(label, label_namespace="label", skip_indexing=True)
            d["good_prefix_length"] = IndexField(
                common_prefix_length(boundary_token_fn(good_prefix), input1), d["input1"]
            )
            d["good_str"] = MetadataField(" ".join(good_prefix))
            d["bad_prefix_length"] = IndexField(
                common_prefix_length(boundary_token_fn(bad_prefix), input2), d["input2"]
            )
            d["bad_str"] = MetadataField(" ".join(bad_prefix))
            d["shared_word_length"] = IndexField(len(shared_word), d["input1"]) 
            d["shared_str"] = MetadataField(" ".join(shared_word))
            d["tagmask"] = MultiLabelField(
                tags, label_namespace="tags", skip_indexing=True, num_labels=len(self.tag_list)
            )
            return Instance(d)

        instances = map(_make_instance, *split)
        return instances


@register_task("blimp-simpleLM", rel_path="blimp")
class BlimpFullSentLMTask(BlimpTask):
    """ Task class for full sentence LM acceptability preference """

    def __init__(self, path, max_seq_len, name, **kw):
        super(BlimpFullSentLMTask, self).__init__(path, max_seq_len, name, **kw)

    def load_data(self):
        """ Load linguistic phenomena benchmark data for simple LM method, each one-prefix
        example includes one shared prefix and the following tokens from the good and bad
        sentences """
        data_file = os.path.join(self.path, "blimp.jsonl")
        data = [json.loads(l) for l in open(data_file, encoding="utf-8").readlines()]
        tag_types = ['field', 'linguistics_term', 'UID']
        sent1s, sent2s, labels, tags = [], [], [], []
        for example in data:
            if not example['simple_LM_method']:
                continue
            sent1s.append(
                tokenize_and_truncate(self._tokenizer_name, example["sentence_good"], self.max_seq_len)
            )
            sent2s.append(
                tokenize_and_truncate(self._tokenizer_name, example["sentence_bad"], self.max_seq_len)
            )
            labels.append(0)
            tags.append([])
            for tag_type in tag_types:
                if tag_type in example:
                    tag_str = "%s__%s" % (tag_type, example[tag_type])
                    tags[-1].append(self.tag_manager(tag_str))
        self.val_data_text = self.test_data_text = self.train_data_text = \
            (sent1s, sent2s, labels, tags)
        self.sentences = self.train_data_text[0] + self.train_data_text[1]
        
        self.tag_list = self.tag_manager.get_tag_list()
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy
        )

        log.info("\tFinished loading blimp simple LM data.")
        return

    def process_split(self, split, indexers, boundary_token_fn):
        def _make_instance(sent1, sent2, label, tags):
            """ from multiple types in one column create multiple fields
            sent1: sentence1, the good one
            sent2: sentence2, the bad one
            label: always 0
            tagmask: which tags this sample has
            """
            d = {}
            d["input1"] = sentence_to_text_field(boundary_token_fn(sent1), indexers)
            d["sent1_str"] = MetadataField(" ".join(sent1))
            d["input2"] = sentence_to_text_field(boundary_token_fn(sent2), indexers)
            d["sent2_str"] = MetadataField(" ".join(sent2))
            d["labels"] = LabelField(label, label_namespace="label", skip_indexing=True)
            d["tagmask"] = MultiLabelField(
                tags, label_namespace="tags", skip_indexing=True, num_labels=len(self.tag_list)
            )
            return Instance(d)

        instances = map(_make_instance, *split)
        return instances


@register_task("npi-cloze-pair", rel_path="NPI")
class NPIClozePairTask(PairClassificationTask):
    """ Task class for cloze test acceptability judgement """

    def __init__(self, path, max_seq_len, name, **kw):
        super(NPIClozePairTask, self).__init__(name, n_classes=2, **kw)        
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None
        self.eval_only_task = True
        
        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]
        
        self.tag_manager = TagManager()
        self.tag_list = None
        self.tag_scorers1 = None
        self.tag_scorers2 = None
        
    def load_data(self):
        """Load the data"""
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "acceptability_cloze_pairs.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            tag2idx_dict={"source": 0, "condition": 4},
            tag_manager=self.tag_manager,
        )
        self.val_data_text = self.test_data_text = self.train_data_text
        self.sentences = self.train_data_text[0] + self.train_data_text[1]

        # Create score for each tag from tag-index dict
        self.tag_list = self.tag_manager.get_tag_list()
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=Correlation, corr_type="matthews"
        )
        self.tag_scorers2 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy
        )

        log.info("\tFinished loading NPI cloze pairs.")
        return

    def process_split(self, split, indexers, boundary_token_fn):
        def _make_instance(input1, input2, labels, tagids):
            """ from multiple types in one column create multiple fields
            input0: shared part (masked form) of both sentence (only used in BERT)
            input1: sentence1
            input2: sentence2
            index: which token is different
            labels: whether sentence1 is the correct sentence
            tagmask: which tags this sample has
            """
            d = {}
            d["input1"] = sentence_to_text_field(boundary_token_fn(input1), indexers)
            d["sent1_str"] = MetadataField(" ".join(input1))
            d["input2"] = sentence_to_text_field(boundary_token_fn(input2), indexers)
            d["sent2_str"] = MetadataField(" ".join(input2))
            mask_index = [i for i in range(len(input1)) if input1[i] != input2[i]][0]
            input0 = [i for i in input1]
            input0[mask_index] = "[MASK]"
            d["input0"] = sentence_to_text_field(boundary_token_fn(input0), indexers)
            d["sent0_str"] = MetadataField(" ".join(input0))
            d["index"] = IndexField(mask_index, d["input1"])
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            d["tagmask"] = MultiLabelField(
                tagids, label_namespace="tags", skip_indexing=True, num_labels=len(self.tag_list)
            )
            return Instance(d)

        instances = map(_make_instance, *split)
        return instances

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


@register_task("npi-minimal-pair", rel_path="NPI")
class NPIMinimalPairTask(PairClassificationTask):
    """ Task class for minimal pair acceptability judgement """

    def __init__(self, path, max_seq_len, name, **kw):
        super(NPIMinimalPairTask, self).__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None
        
        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorer3 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2, self.scorer3]
        
        self.tag_manager = TagManager()
        self.tag_list = None
        self.tag_scorers1 = None
        self.tag_scorers2 = None
        self.tag_scorers3 = None

    def load_data(self):
        """Load the data"""
        self.train_data_text = load_tsv(
            self._tokenizer_name,
            os.path.join(self.path, "acceptability_minimal_pairs.tsv"),
            max_seq_len=self.max_seq_len,
            s1_idx=1,
            s2_idx=2,
            label_idx=3,
            tag2idx_dict={"source": 0, "condition": 4},
            tag_manager=self.tag_manager,
        )
        self.val_data_text = self.test_data_text = self.train_data_text
        # Create score for each tag from tag-index dict
        self.tag_list = self.tag_manager.get_tag_list()
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=Correlation, corr_type="matthews"
        )
        self.tag_scorers2 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy
        )
        self.tag_scorers3 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy
        )

        log.info("\tFinished loading NPI minimal pairs.")
        return

    def process_split(self, split, indexers, boundary_token_fn):
        def _make_instance(input1, input2, labels, tagids):
            """ from multiple types in one column create multiple fields
            input1: sentence1
            input2: sentence2
            labels: whether sentence1 is the correct sentence
            tagmask: which tags this sample has
            """
            d = {}
            d["input1"] = sentence_to_text_field(boundary_token_fn(input1), indexers)
            d["sent1_str"] = MetadataField(" ".join(input1))
            d["input2"] = sentence_to_text_field(boundary_token_fn(input2), indexers)
            d["sent2_str"] = MetadataField(" ".join(input2))
            d["labels"] = LabelField(labels, label_namespace="labels", skip_indexing=True)
            d["tagmask"] = MultiLabelField(
                tagids, label_namespace="tagids", skip_indexing=True, num_labels=len(self.tag_list)
            )

            return Instance(d)

        instances = map(_make_instance, *split)
        return instances

    def update_metrics(self, logits, labels, tagmask=None):
        logits, labels = logits.detach(), labels.detach()
        logits_relative = logits[:, : self.n_classes]
        _, preds = logits_relative.max(dim=1)
        self.scorer1(preds, labels)
        self.scorer2(logits_relative, labels)
        self.scorer3(logits, labels)
        if tagmask is not None:
            update_subset_scorers(self.tag_scorers1, preds, labels, tagmask)
            update_subset_scorers(self.tag_scorers2, logits_relative, labels, tagmask)
            update_subset_scorers(self.tag_scorers3, logits, labels, tagmask)
        return

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""

        collected_metrics = {
            "mcc": self.scorer1.get_metric(reset),
            "accuracy": self.scorer2.get_metric(reset),
            "accuracy_strict": self.scorer3.get_metric(reset),
        }
        collected_metrics.update(
            collect_subset_scores(self.tag_scorers1, "mcc", self.tag_list, reset)
        )
        collected_metrics.update(
            collect_subset_scores(self.tag_scorers2, "accuracy", self.tag_list, reset)
        )
        collected_metrics.update(
            collect_subset_scores(self.tag_scorers3, "accuracy_strict", self.tag_list, reset)
        )
        return collected_metrics


def common_prefix_length(sent1, sent2):
    min_length = min(len(sent1), len(sent2))
    for i, (x, y) in enumerate(zip(sent1[:min_length], sent2[:min_length])):
        if x != y:
            return i
    return min_length