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

    def _get_num_labels(self) -> int:
        return self.vocab.get_vocab_size(namespace=self.label_namespace)

    def _label_ids_to_khot(self, label_ids: List[int]) -> np.ndarray:
        arr = np.zeros(self._get_num_labels(), dtype=np.int32)
        arr[label_ids] = 1
        return arr

    def _get_label(self, i: int) -> str:
        return self.vocab.get_token_from_index(i, namespace=self.label_namespace)

    def __init__(self, vocab: Vocabulary, records: Iterable[Dict],
                 label_namespace=None):
        self.vocab = vocab
        self.label_namespace = label_namespace
        self.all_labels = [self._get_label(i)
                           for i in range(self._get_num_labels())]

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

        # Placeholders, will compute later if requested.
        # Use non-underscore versions to access via propert getters.
        self._target_df_wide = None  # wide-form targets (expanded)
        self._target_df_long = None  # long-form targets (melted by label)

    def _make_wide_target_df(self):
        print("Generating wide-form target DataFrame. May be slow...")
        # Expand labels to columns
        expanded_y_true = self.target_df['label.khot'].apply(pd.Series)
        expanded_y_true.columns = ["label.true." + l for l in self.all_labels]
        expanded_y_pred = self.target_df['preds.proba'].apply(pd.Series)
        expanded_y_pred.columns = ["preds.proba." + l for l in self.all_labels]
        wide_df = pd.concat([self.target_df, expanded_y_true, expanded_y_pred],
                            axis='columns')
        DROP_COLS = ["preds.proba", "label", "label.ids", "label.khot"]
        wide_df.drop(labels=DROP_COLS, axis=1, inplace=True)
        return wide_df

    @property
    def target_df_wide(self):
        """Target df in wide form. Compute only if requested."""
        if self._target_df_wide is None:
            self._target_df_wide = self._make_wide_target_df()
        return self._target_df_wide

    def _make_long_target_df(self):
        wide_df = self.target_df_wide
        print("Generating long-form target DataFrame. May be slow...")
        # Melt to wide, using dummy 'index' column as unique key.
        # All cols not starting with stubnames are kept as id_vars.
        long_df = pd.wide_to_long(wide_df.reset_index(),
                                  i=['index'], j="label",
                                  stubnames=["label.true", "preds.proba"],
                                  sep=".",
                                  suffix=r"\w+")
        long_df.sort_values("idx", inplace=True)  # Sort by example idx
        long_df.reset_index(inplace=True)         # Remove multi-index
        long_df.drop("index", axis=1, inplace=True)  # Drop dummy index, but keep 'label'
        long_df.sort_index(axis=1, inplace=True)  # Sort columns alphabetically
        return long_df

    @property
    def target_df_long(self):
        """Target df in long form (one row per label, per target).
        Compute only if requested.
        """
        if self._target_df_long is None:
            self._target_df_long = self._make_long_target_df()
        return self._target_df_long

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

