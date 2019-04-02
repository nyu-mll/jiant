"""Task definitions for NPI licensing tasks."""
import json
import logging as log
import math
import os

from allennlp.training.metrics import Average
from allennlp.data.token_indexers import SingleIdTokenIndexer

# Fields for instance processing
from allennlp.data import Instance, Token

from ..utils.utils import truncate
from ..utils.data_loaders import process_sentence

from typing import Iterable, Sequence, List, Dict, Any, Type

from .tasks import SingleClassificationTask
from .tasks import sentence_to_text_field, atomic_tokenize
from .tasks import UNK_TOK_ALLENNLP, UNK_TOK_ATOMIC
from .registry import register_task


@register_task('npi', rel_path='npi/')
class NPIAcceptabilityTask(SingleClassificationTask):
    ''' Task class for NPI Acceptability Judgement.  '''

    def load_data(self, path, max_seq_len):
        ''' Load data '''
        import pdb;pdb.set_trace()
        tr_data = load_tsv(self._tokenizer_name, os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=0, s2_idx=None, label_idx=1, skip_rows=1)
        val_data = load_tsv(self._tokenizer_name, os.path.join(path, 'dev.tsv'), max_seq_len,
                            s1_idx=0, s2_idx=None, label_idx=1, skip_rows=1)
        te_data = load_tsv(self._tokenizer_name, os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=None, has_labels=False, return_indices=True, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SST data.")