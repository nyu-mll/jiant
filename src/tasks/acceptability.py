'''Define the tasks and code for loading their data.

- As much as possible, following the existing task hierarchy structure.
- When inheriting, be sure to write and call load_data.
- Set all text data as an attribute, task.sentences (List[List[str]])
- Each task's val_metric should be name_metric, where metric is returned by
get_metrics(): e.g. if task.val_metric = task_name + "_accuracy", then
task.get_metrics() should return {"accuracy": accuracy_val, ... }
'''
import codecs
import collections
import copy
import logging as log
import os

import numpy as np
import torch

from allennlp.training.metrics import CategoricalAccuracy
from ..allennlp_mods.correlation import Correlation
from allennlp.data import vocabulary
from .registry import register_task

# Fields for instance processing
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, LabelField, MultiLabelField, \
    MetadataField, IndexField

from ..utils.data_loaders import load_tsv, get_tag_list, BERT_MASK_TOK
from .tasks import Task, sentence_to_text_field, SingleClassificationTask
from .tasks import create_subset_scorers, update_subset_scorers, collect_subset_scores

from typing import Iterable, Sequence, List, Dict, Any, Type


@register_task('npi_pair_frozen', rel_path='NPI')
class NPIMinimalPairFrozenTask(Task):
    ''' Task class for frozen minimal pair (cloze) acceptability judgement '''
    def __init__(self, path, max_seq_len, name, **kw):
        super(NPIMinimalPairFrozenTask, self).__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self.n_classes = 2

        s`elf.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.val_metric = "%s_mcc" % self.name
        self.val_metric_decreases = False
        self.scorer1 = Correlation("matthews")
        self.scorer2 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2]

    def load_data(self):
        '''Load the data'''
        tag_vocab = vocabulary.Vocabulary(counter=None)
        self.train_data_text = load_tsv(
            self._tokenizer_name, os.path.join(self.path, "acceptability_cloze_pairs.tsv"),
            max_seq_len=self.max_seq_len, s1_idx=1, s2_idx=2, label_idx=3,
            tag2idx_dict={'source': 0, 'condition': 4}, tag_vocab=tag_vocab
        )
        self.val_data_text = self.test_data_text = self.train_data_text
        self.sentences = self.train_data_text[0] + self.train_data_text[1]

        # Create score for each tag from tag-index dict
        self.tag_list = get_tag_list(tag_vocab)
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list),
            scorer_type=Correlation,
            corr_type="matthews")
        self.tag_scorers2 = create_subset_scorers(
            count=len(self.tag_list),
            scorer_type=CategoricalAccuracy)

        log.info("\tFinished loading CoLA cloze pairs.")
        return

    def process_split(self, split, indexers):
        def _make_instance(input1, input2, labels, tagids):
            ''' from multiple types in one column create multiple fields
            input0: shared part (masked form) of both sentence (only used in BERT)
            input1: sentence1
            input2: sentence2
            index: which token is different
            labels: is
            tagmask: which tag the pair has, in this dataset, the source of data
            '''
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["sent1_str"] = MetadataField(" ".join(input1[1:-1]))
            d["input2"] = sentence_to_text_field(input2, indexers)
            d["sent2_str"] = MetadataField(" ".join(input2[1:-1]))
            if self.name == "npi_pair_frozen":
                mask_index = [i for i in range(len(input1)) if input1[i] != input2[i]][0]
                input0 = [i for i in input1]
                input0[mask_index] = BERT_MASK_TOK
                d["input0"] = sentence_to_text_field(input0, indexers)
                d["sent0_str"] = MetadataField(" ".join(input0[1:-1]))
                d["index"] = IndexField(mask_index, d["input1"])
            d["labels"] = LabelField(labels, label_namespace="labels",
                                     skip_indexing=True)
            d["tagmask"] = MultiLabelField(tagids, label_namespace="tagids",
                                           skip_indexing=True, num_labels=len(self.tag_list))

            return Instance(d)

        instances = map(_make_instance, *split)
        return instances  # lazy iterator

    def update_metrics(self, logits, labels, tagmask=None):
        logits, labels = logits.detach(), labels.detach()
        logits_relative = logits[:, :self.n_classes]
        _, preds = logits_relative.max(dim=1)
        self.scorer1(preds, labels)
        self.scorer2(logits_relative, labels)
        if tagmask is not None:
            update_subset_scorers(self.tag_scorers1, preds, labels, tagmask)`
            update_subset_scorers(self.tag_scorers2, logits_relative, labels, tagmask)
        return

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''

        collected_metrics = {
            'mcc': self.scorer1.get_metric(reset),
            'accuracy': self.scorer2.get_metric(reset)}
        collected_metrics.update(
            collect_subset_scores(
                self.tag_scorers1, 'mcc', self.tag_list, reset
            ))
        collected_metrics.update(
            collect_subset_scores(
                self.tag_scorers2, 'accuracy', self.tag_list, reset
            ))
        return collected_metrics


@register_task('npi_pair_tuned', rel_path='NPI')
class NPIMinimalPairTunedTask(NPIMinimalPairFrozenTask):
    ''' Task class for tuned minimal pair acceptability judgement '''
    def __init__(self, path, max_seq_len, name, **kw):
        super(NPIMinimalPairTunedTask, self).__init__(path, max_seq_len, name, **kw)
        self.scorer3 = CategoricalAccuracy()
        self.scorers = [self.scorer1, self.scorer2, self.scorer3]

    def load_data(self):
        '''Load the data'''
        tag_vocab = vocabulary.Vocabulary(counter=None)
        self.train_data_text = load_tsv(
            self._tokenizer_name, os.path.join(self.path, "acceptability_minimal_pairs.tsv"),
            max_seq_len=self.max_seq_len, s1_idx=1, s2_idx=2, label_idx=3,
            tag2idx_dict={'source': 0, 'condition': 4}, tag_vocab=tag_vocab
        )
        self.val_data_text = self.test_data_text = self.train_data_text
        # Create score for each tag from tag-index dict
        self.tag_list = get_tag_list(tag_vocab)
        self.tag_scorers1 = create_subset_scorers(
            count=len(self.tag_list),
            scorer_type=Correlation,
            corr_type="matthews")
        self.tag_scorers2 = create_subset_scorers(
            count=len(self.tag_list),
            scorer_type=CategoricalAccuracy)
        self.tag_scorers3 = create_subset_scorers(
            count=len(self.tag_list),
            scorer_type=CategoricalAccuracy)

        log.info("\tFinished loading CoLA minimal pairs.")
        return

    def update_metrics(self, logits, labels, tagmask=None):
        logits, labels = logits.detach(), labels.detach()
        logits_relative = logits[:, :self.n_classes]
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
        '''Get metrics specific to the task'''

        collected_metrics = {
            'mcc': self.scorer1.get_metric(reset),
            'accuracy': self.scorer2.get_metric(reset),
            'accuracy_strict': self.scorer3.get_metric(reset)}
        collected_metrics.update(
            collect_subset_scores(
                self.tag_scorers1, 'mcc', self.tag_list, reset
            ))
        collected_metrics.update(
            collect_subset_scores(
                self.tag_scorers2, 'accuracy', self.tag_list, reset
            ))
        collected_metrics.update(
            collect_subset_scores(
                self.tag_scorers3, 'accuracy_strict', self.tag_list, reset
            ))
        return collected_metrics


@register_task('cola-analysis', rel_path='CoLA/')
class CoLAAnalysisTask(SingleClassificationTask):
    """
    cola-analysis dataset only tagged the dev set of cola
    when using this dataset, the model need to have a cola classifier available
    """
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

    def load_data(self):
        '''Load the data'''
        # Load data from tsv
        tag_vocab = vocabulary.Vocabulary(counter=None)
        val_data = load_tsv(
            tokenizer_name=self._tokenizer_name,
            data_file=os.path.join(self.path, "dev_analysis.tsv"), max_seq_len=self.max_seq_len,
            s1_idx=3, s2_idx=None, label_idx=2, skip_rows=1, tag2idx_dict={
                'Domain': 1, 'Simple': 4, 'Pred': 5, 'Adjunct': 6, 'Arg Types': 7, 'Arg Altern': 8,
                'Imperative': 9, 'Binding': 10, 'Question': 11, 'Comp Clause': 12, 'Auxillary': 13,
                'to-VP': 14, 'N, Adj': 15, 'S-Syntax': 16, 'Determiner': 17, 'Violations': 18
            }, tag_vocab=tag_vocab)
        self.train_data_text = val_data
        self.val_data_text = val_data
        self.test_data_text = val_data
        self.sentences = self.train_data_text[0] + self.val_data_text[0]
        # Create score for each tag from tag-index dict
        self.tag_list = get_tag_list(tag_vocab)
        self.tag_scorers1 = create_subset_scorers(
            count=len(
                self.tag_list),
            scorer_type=Correlation,
            corr_type="matthews")
        self.tag_scorers2 = create_subset_scorers(
            count=len(self.tag_list), scorer_type=CategoricalAccuracy)

        log.info("\tFinished loading CoLA sperate domain.")

    def process_split(self, split, indexers):
        def _make_instance(input1, input2, labels, tagids):
            ''' from multiple types in one column create multiple fields '''
            d = {}
            d["input1"] = sentence_to_text_field(input1, indexers)
            d["sent1_str"] = MetadataField(" ".join(input1[1:-1]))
            d["labels"] = LabelField(labels, label_namespace="labels",
                                     skip_indexing=True)
            d["tagmask"] = MultiLabelField(tagids, label_namespace="tagids",
                                           skip_indexing=True, num_labels=len(self.tag_list))
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
        '''Get metrics specific to the task'''

        collected_metrics = {
            'mcc': self.scorer1.get_metric(reset),
            'accuracy': self.scorer2.get_metric(reset)}
        collected_metrics.update(
            collect_subset_scores(
                self.tag_scorers1,
                'mcc',
                self.tag_list,
                reset))
        collected_metrics.update(
            collect_subset_scores(
                self.tag_scorers2,
                'accuracy',
                self.tag_list,
                reset))
        return collected_metrics


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
