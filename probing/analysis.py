##
# Helper libraries for #datascience on Edge-Probing data.

import sys
import os
import json

import pandas as pd
import numpy as np

from src import utils
from allennlp.data import Vocabulary

from typing import Iterable, Dict, List

def _get_nested_vals(record, outer_key):
	return {f"{outer_key}.{key}": value
			for key, value in record.get(outer_key, {}).items()}

class Predictions(object):
    """Container class to manage a set of predictions from the Edge Probing
    model. Recommended usage:

    preds = analysis.Predictions.from_run("/path/to/exp/run",
                                          "edges-srl-conll2005",
                                          "val")

    # preds has the following fields:
    preds.vocab       # allennlp.data.Vocabulary object
    preds.example_df  # DataFrame of example info (sentence text)
    preds.target_df   # DataFrame of target info (spans, labels,
                      #   predicted scores, etc.)

    """
    def _split_and_flatten_records(self, records: Iterable[Dict]):
        ex_records = []  # long-form example records, minus targets
        tr_records = []  # long-form target records with 'idx' column
        for idx, r in enumerate(records):
            d = {'text': r['text'], 'idx': idx}
            d.update(_get_nested_vals(r, 'info'))
            d.update(_get_nested_vals(r, 'preds'))
            ex_records.append(d)

            for t in r['targets']:
                d = {'label': utils.wrap_singleton_string(t['label']),
                     'idx': idx}
                if 'span1' in t:
                    d['span1'] = t['span1']
                if 'span2' in t:
                    d['span2'] = t['span2']
                d.update(_get_nested_vals(t, 'info'))
                d.update(_get_nested_vals(t, 'preds'))
                tr_records.append(d)
        return ex_records, tr_records

    def _labels_to_ids(self, labels: List[str]) -> List[int]:
        return [self.vocab.get_token_index(l, namespace=self.label_namespace)
                for l in labels]

    def _label_ids_to_khot(self, label_ids: List[int]) -> np.ndarray:
        vlen = self.vocab.get_vocab_size(namespace=self.label_namespace)
        arr = np.zeros(vlen, dtype=np.int32)
        arr[label_ids] = 1
        return arr

    def __init__(self, vocab: Vocabulary, records: Iterable[Dict],
                 label_namespace=None):
        self.vocab = vocab
        self.label_namespace = label_namespace

        ex_records, tr_records = self._split_and_flatten_records(records)
        self.example_df = pd.DataFrame.from_records(ex_records)
        self.example_df.set_index('idx', inplace=True, drop=False)
        self.target_df = pd.DataFrame.from_records(tr_records)

        # Apply indexing to labels
        self.target_df['label.ids'] = self.target_df['label'].map(
                                            self._labels_to_ids)
        # Convert labels to k-hot to align to predictions
        self.target_df['label.khot'] = self.target_df['label.ids'].map(
                                            self._label_ids_to_khot)


    @classmethod
    def from_run(cls, run_dir: str, task_name: str, split_name: str):
        # Load vocabulary
        exp_dir = os.path.dirname(run_dir.rstrip("/"))
        vocab_path = os.path.join(exp_dir, "vocab")
        vocab = Vocabulary.from_files(vocab_path)
        label_namespace = f"{task_name}_labels"

        # Load predictions
        preds_file = os.path.join(run_dir, f"{task_name}_{split_name}.json")
        return cls(vocab, utils.load_json_data(preds_file),
                   label_namespace=label_namespace)

