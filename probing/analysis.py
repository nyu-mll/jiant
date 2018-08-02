##
# Helper libraries for #datascience on Edge-Probing data.

import sys
import os
import json
import collections
import itertools

import pandas as pd
import numpy as np
from sklearn import metrics

from src import utils
from allennlp.data import Vocabulary

from typing import Iterable, Dict, List, Tuple

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
        print("Generating wide-form target DataFrame. May be slow... ", end="")
        # Expand labels to columns
        expanded_y_true = self.target_df['label.khot'].apply(pd.Series)
        expanded_y_true.columns = ["label.true." + l for l in self.all_labels]
        expanded_y_pred = self.target_df['preds.proba'].apply(pd.Series)
        expanded_y_pred.columns = ["preds.proba." + l for l in self.all_labels]
        wide_df = pd.concat([self.target_df, expanded_y_true, expanded_y_pred],
                            axis='columns')
        DROP_COLS = ["preds.proba", "label", "label.ids", "label.khot"]
        wide_df.drop(labels=DROP_COLS, axis=1, inplace=True)
        print("Done!")
        return wide_df

    @property
    def target_df_wide(self):
        """Target df in wide form. Compute only if requested."""
        if self._target_df_wide is None:
            self._target_df_wide = self._make_wide_target_df()
        return self._target_df_wide

    def _make_long_target_df(self):
        wide_df = self.target_df_wide
        print("Generating long-form target DataFrame. May be slow... ", end="")
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
        print("Done!")
        return long_df

    @property
    def target_df_long(self):
        """Target df in long form (one row per label, per target).
        Compute only if requested.
        """
        if self._target_df_long is None:
            self._target_df_long = self._make_long_target_df()
        return self._target_df_long

    def score_by_label(self) -> pd.DataFrame:
        """Compute metrics for each label."""
        wide_df = self.target_df_wide
        records = []
        for label in self.all_labels:
            y_true = wide_df['label.true.' + label]
            y_pred = wide_df['preds.proba.' + label] >= 0.5
            f1_score = metrics.f1_score(y_true=y_true, y_pred=y_pred)
            acc_score = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
            true_count = sum(y_true)
            pred_count = sum(y_pred)
            records.append(dict(label=label, f1_score=f1_score,
                                acc_score=acc_score,
                                true_count=true_count,
                                pred_count=pred_count))
        return pd.DataFrame.from_records(records)

    @classmethod
    def from_run(cls, run_dir: str, task_name: str, split_name: str):
        # Load vocabulary
        exp_dir = os.path.dirname(run_dir.rstrip("/"))
        vocab_path = os.path.join(exp_dir, "vocab")
        print("Loading vocabulary from %s" % vocab_path)
        vocab = Vocabulary.from_files(vocab_path)
        label_namespace = f"{task_name}_labels"

        # Load predictions
        preds_file = os.path.join(run_dir, f"{task_name}_{split_name}.json")
        print("Loading predictions from %s" % preds_file)
        return cls(vocab, utils.load_json_data(preds_file),
                   label_namespace=label_namespace)


class Comparison(object):
    """Container class to represent a pair of experiments."""

    def __init__(self, base: Predictions, expt: Predictions,
                 label_filter=lambda label: True):
        assert len(base.all_labels) == len(expt.all_labels)
        assert len(base.example_df) == len(expt.example_df)
        assert len(base.target_df)  == len(expt.target_df)

        self.base = base
        self.expt = expt

        # Score & compare
        print("Scoring base run...")
        self.base_scores = base.score_by_label()
        self.base_scores['run'] = "base"
        print("Scoring expt run...")
        self.expt_scores = expt.score_by_label()
        self.expt_scores['run'] = "expt"
        print("Done scoring!")

        _mask = self.base_scores['label'].map(label_filter)
        self.base_scores = self.base_scores[_mask]
        _mask = self.expt_scores['label'].map(label_filter)
        self.expt_scores = self.expt_scores[_mask]

        self.long_scores = pd.concat([self.base_scores,
                                      self.expt_scores])
        # Wide-form scores for direct comparison
        df = pd.merge(self.base_scores, self.expt_scores,
                      on=["label", "true_count"],
                      suffixes=("_base", "_expt"))
        df['abs_diff_f1'] = (df["f1_score_expt"] - df["f1_score_base"])
        # Compute relative error reduction
        df['rel_diff_f1'] = (df['abs_diff_f1'] / (1 - df["f1_score_base"]))
        # Net diffs, computed from accuracy
        df['net_diffs'] = (np.abs(df['acc_score_expt'] - df['acc_score_base'])
                           * len(base.target_df)).astype(np.int32)
        df['base_headroom'] = ((1 - df['acc_score_base'])
                               * len(base.target_df)).astype(np.int32)
        df['expt_headroom'] = ((1 - df['acc_score_expt'])
                               * len(expt.target_df)).astype(np.int32)
        self.wide_scores = df


    def plot_scores(self, task_name, metric="f1", sort_field="expt_headroom",
                    sort_ascending=False, row_height=400):
        import bokeh
        import bokeh.plotting as bp

        _SCORE_COL = f"{metric:s}_score"
        _ABS_DIFF_COL = f"abs_diff_{metric:s}"
        _REL_DIFF_COL = f"rel_diff_{metric:s}"

        # Long-form data, for paired bar chart.
        # TODO: switch to using wide-form for all?
        long_df = self.long_scores.copy()
        long_df['row_key'] = list(zip(long_df['label'], long_df['run']))
        long_df['fmt_score'] = long_df[_SCORE_COL].map(
            lambda s: "{:.02f}".format(s)
        )
        long_ds = bokeh.models.ColumnDataSource(data=long_df)

        # Wide-form data, for counts & comparison metrics.
        wide_df = self.wide_scores.copy()
        wide_df['_diff_label_offset'] = 10 * np.sign(wide_df[_ABS_DIFF_COL])
        wide_df['_count_diff_offset'] = -1.25 * wide_df['_diff_label_offset']

        # Formatting label text
        wide_df['_net_diff_count_label_text'] = wide_df['net_diffs'].map(
            lambda s: "~ {:d} ex".format(s) if s else "")
        wide_df['_abs_diff_label_text'] = wide_df[_ABS_DIFF_COL].map(
            lambda s: "{:.02f}".format(s))
        wide_df['_rel_diff_label_text'] = wide_df[_REL_DIFF_COL].map(
            lambda s: "({:.02f})".format(s) if s else "")
        wide_ds = bokeh.models.ColumnDataSource(data=wide_df)

        # Prepare shared categorical axis
        runs = sorted(long_df['run'].unique())
        labels = wide_df.sort_values(by=sort_field,
                                     ascending=sort_ascending)['label']
        categories = list(itertools.product(labels, runs))

        palette = bokeh.palettes.Spectral6
        fill_cmap = bokeh.transform.factor_cmap('run', palette, runs)
        width = 35*len(categories)
        tools = 'xwheel_zoom,xwheel_pan,xpan,save,reset'

        # Top plot: score bars
        factor_range = bokeh.models.FactorRange(*categories,
                                                range_padding=0.5,
                                                range_padding_units='absolute')
        p1 = bp.figure(title=f"Performance by label ({task_name})",
                       x_range=factor_range, y_range=[0,1],
                       width=width, height=row_height, tools=tools)
        p1.vbar(x='row_key', top=_SCORE_COL, width=0.95,
               fill_color=fill_cmap, line_color=None,
               source=long_ds)
        label_kw = dict(text_align="right", text_baseline="middle", y_offset=-3,
                        text_font_size="11pt", angle=90, angle_units='deg')
        score_labels = bokeh.models.LabelSet(x='row_key', y=_SCORE_COL,
                                             text="fmt_score",
                                             source=long_ds, **label_kw)
        p1.add_layout(score_labels)
        p1.xaxis.major_label_orientation = 1
        p1.yaxis.bounds = (0,1)

        # Second plot: absolute diffs
        p2 = bp.figure(title=f"Absolute and (Relative) diffs by label ({task_name})",
                       x_range=p1.x_range, width=width, height=row_height,
                       tools=tools)
        p2.vbar(x='label', top='rel_diff_f1', width=1.90,
                fill_color='DarkRed', fill_alpha=0.20,
                line_color=None,
                source=wide_ds)
        p2.vbar(x='label', top='abs_diff_f1', width=1.90,
                fill_color='DarkRed', line_color=None,
                source=wide_ds)
        label_kw = dict(text_align="center", text_baseline="middle",
                        y_offset="_diff_label_offset", source=wide_ds)
        delta_labels = bokeh.models.LabelSet(x='label', y="abs_diff_f1",
                                             text="_abs_diff_label_text",
                                             **label_kw)
        p2.add_layout(delta_labels)
        rel_labels = bokeh.models.LabelSet(x='label', y="rel_diff_f1",
                                             text="_rel_diff_label_text",
                                             **label_kw)
        p2.add_layout(rel_labels)
        count_labels = bokeh.models.LabelSet(x='label', y=0,
                                             y_offset="_count_diff_offset",
                                             text="_net_diff_count_label_text",
                                             text_align="center", text_baseline="middle",
                                             text_font_style="italic", text_color="gray",
                                             text_font_size="10pt",
                                             source=wide_ds)
        p2.add_layout(count_labels)

        p2.y_range.start = -1.10
        p2.y_range.end = 1.10
        p2.yaxis.bounds = (-1,1)
        # Hacky: Hide category labels, not needed on this plot.
        p2.xaxis.major_label_text_color = None
        p2.xaxis.major_label_text_font_size = "0pt"
        p2.xaxis.major_tick_line_color = None


        # Bottom plot: count bars
        p3 = bp.figure(title=f"Counts by label ({task_name})",
                       x_range=p1.x_range, width=width, height=row_height,
                       tools=tools)
        p3.vbar(x='label', top='true_count', width=1.90,
                fill_color='orange', line_color=None,
                source=wide_ds)
        label_kw = dict(text_align="center", text_baseline="top", y_offset=-5)
        count_labels = bokeh.models.LabelSet(x='label', y="true_count",
                                             text="true_count",
                                             source=wide_ds, **label_kw)
        p3.add_layout(count_labels)
        p3.y_range.flipped = True
        p3.y_range.end = 0
        p3.y_range.range_padding = 0.20
        # Hacky: Hide category labels, not needed on this plot.
        p3.xaxis.major_label_text_color = None
        p3.xaxis.major_label_text_font_size = "0pt"
        p3.xaxis.major_tick_line_color = None

        # Fix labels for SPR case, labels are long
        if max(map(len, labels)) > 10:
            # Top plot: rotate labels, add height.
            p1.xaxis.group_label_orientation = np.pi/2
            p1.plot_height += 150
            # Middle plot: hide labels.
            p2.xaxis.group_text_color = None
            p2.xaxis.group_text_font_size = "0pt"
            # Bottom plot: rotate labels, add height.
            p3.xaxis.group_label_orientation = np.pi/2
            p3.plot_height += 75

        # Create plot layout.
        plots = bokeh.layouts.gridplot([p1, p2, p3], ncols=1,
                                      toolbar_location="left",
                                      merge_tools=True,
                                      sizing_mode="fixed")
        header = bokeh.models.Div(
            text=f"<h1>{task_name} sorted by '{sort_field}'</h1>",
            width=600)
        return bokeh.layouts.column(header, plots)
