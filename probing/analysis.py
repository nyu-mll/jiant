##
# Helper libraries for #datascience on Edge-Probing data.

import collections
import io
import itertools
import json
import logging as log
import os
import re
import sys

from allennlp.data import Vocabulary
from bokeh import palettes
from sklearn import metrics
import numpy as np
import pandas as pd

from jiant.utils import utils

from typing import Dict, Iterable, List, Tuple

##
# Task list for stable ordering, and human-friendly display names.
TASK_TO_DISPLAY_NAME = collections.OrderedDict(
    [
        ("pos-ontonotes", "Part-of-Speech"),
        ("nonterminal-ontonotes", "Constituents"),
        ("dep-labeling-ewt", "Dependencies"),  # old task name
        ("dep-ud-ewt", "Dependencies"),
        ("ner-ontonotes", "Entities"),
        ("srl-conll2012", "SRL"),  # old task name
        ("srl-ontonotes", "SRL"),
        ("coref-ontonotes-conll", "OntoNotes Coref."),  # old task name
        ("coref-ontonotes", "OntoNotes Coref."),
        ("spr1", "SPR1"),
        ("spr2", "SPR2"),
        ("dpr", "Winograd Coref."),
        ("rel-semeval", "Relations (SemEval)"),
    ]
)
TASKS = list(TASK_TO_DISPLAY_NAME.keys())


def task_sort_key(candidate):
    """Generate a stable sort key for a task, with optional suffixes."""
    for i, name in enumerate(TASKS):
        if candidate.startswith(name):
            return (i, candidate)
    return (len(TASKS), candidate)


def clean_task_name(task_name):
    """Return a cleaned version of a jiant task name."""
    c1 = re.sub(r"^edges-", "", task_name)
    c2 = re.sub(r"-openai$", "", c1)  # legacy, for old -openai versions of tasks
    return c2


def make_display_name(task, label=None):
    display_task = TASK_TO_DISPLAY_NAME[task]
    if label in {"_micro_avg_", "1", None}:
        return display_task
    elif label == "_clean_micro_":
        return f"{display_task} (all)"
    elif label == "_core_":
        return f"{display_task} (core)"
    elif label == "_non_core_":
        return f"{display_task} (non-core)"
    else:
        clean_label = label.strip("_")
        return f"{display_task} ({clean_label})"


# See https://bokeh.pydata.org/en/latest/docs/reference/palettes.html
_clist = palettes.Category20[20]
# Experiment type list for stable ordering
# These correspond to the convention in scripts/edges/exp_fns.sh
# and scripts/edges/kubernetes_run_all.sh for naming experiments.
exp_types_clist_idx = [
    ("glove", 2),  # orange
    ("cove", 6),  # deep red
    ("elmo-chars", 18),  # aqua
    ("elmo-ortho", 8),  # purple
    ("elmo-full", 0),  # blue
    ("openai-lex", 16),  # olive
    ("openai-cat", 4),  # green
    ("openai-mix", 12),  # pink
    ("openai", 4),  # green
    ("openai-bwb", 12),  # pink
    ("train-chars", 10),  # brown
]
# Add BERT experiments; all the same colors.
for bert_name in ["base-uncased", "base-cased", "large-uncased", "large-cased"]:
    exp_types_clist_idx.append((f"bert-{bert_name}-lex", 16))  # olive
    exp_types_clist_idx.append((f"bert-{bert_name}-cat", 4))  # green
    exp_types_clist_idx.append((f"bert-{bert_name}-mix", 12))  # pink
    exp_types_clist_idx.append((f"bert-{bert_name}-at", 6))  # deep red

exp_types_colored = collections.OrderedDict()
# Use lighter versions for base model, darker for CNN
for k, v in exp_types_clist_idx:
    exp_types_colored[k] = _clist[v + 1]
    exp_types_colored[k + "-cnn1"] = _clist[v]
    exp_types_colored[k + "-cnn2"] = _clist[v]

EXP_TYPES, EXP_PALETTE = zip(*exp_types_colored.items())


def exp_type_sort_key(candidate):
    """Generate a stable sort key for an experiment type, with optional suffixes."""
    exp_type = candidate.split(" ", 1)[0]
    m = re.match(r"(.*)-\d+$", exp_type)
    if m:
        exp_type = m.group(1)
    return (EXP_TYPES.index(exp_type), candidate)


def _parse_exp_name(exp_name):
    m = re.match(r"([a-z-]+)([-_](\d+))?-edges-([a-z-]+)", exp_name)
    assert m is not None, f"Unable to parse run name: {exp_name}"
    prefix, _, num, task = m.groups()
    return prefix, num, task


def get_exp_type(exp_name):
    return _parse_exp_name(exp_name)[0]


def get_layer_num(exp_name):
    return _parse_exp_name(exp_name)[1]


##
# Predicates for filtering and aggregation


def is_core_role(label):
    return re.match(r"^ARG[0-5A]$", label) is not None


def is_non_core_role(label):
    return re.match(r"^ARGM(-.+)?$", label) is not None


def is_core_or_noncore(label):
    return is_core_role(label) or is_non_core_role(label)


def is_srl_task(task):
    return task.startswith("srl-")


def is_coref_task(task):
    return task.startswith("coref-")


def is_relation_task(task):
    return task.startswith("rel-")


def is_positive_relation(label):
    return (not label.startswith("_")) and (label != "no_relation") and (label != "Other")


def spans_intersect(a, b):
    if a[0] <= b[0] and b[0] < a[1]:
        return True
    if b[0] <= a[0] and a[0] < b[1]:
        return True
    return False


##
# Scoring helpers


def harmonic_mean(a, b):
    return 2 * a * b / (a + b)


def score_from_confusion_matrix(df):
    """Score a DataFrame in-place, computing metrics for each row based on confusion matricies."""
    assert "tn_count" in df.columns  # true negatives
    assert "fp_count" in df.columns  # false positives
    assert "fn_count" in df.columns  # false negatives
    assert "tp_count" in df.columns  # true negatives

    df["pred_pos_count"] = df.tp_count + df.fp_count
    df["true_pos_count"] = df.tp_count + df.fn_count
    df["total_count"] = df.tp_count + df.tn_count + df.fp_count + df.fn_count

    # NOTE: this overwrites any _macro_avg_ rows by recomputing the micro-average!
    df["accuracy"] = (df.tp_count + df.tn_count) / df.total_count
    df["precision"] = df.tp_count / df.pred_pos_count
    df["recall"] = df.tp_count / df.true_pos_count
    df["f1_score"] = harmonic_mean(df.precision, df.recall).fillna(0)

    # Approximate error intervals using normal approximation
    # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
    z = 1.96  # 95% confidence
    df["accuracy_errn95"] = z * (df.accuracy * (1 - df.accuracy) / df.total_count).map(np.sqrt)
    df["precision_errn95"] = z * (df.precision * (1 - df.precision) / df.pred_pos_count).map(
        np.sqrt
    )
    df["recall_errn95"] = z * (df.recall * (1 - df.recall) / df.true_pos_count).map(np.sqrt)
    # This probably isn't the right way to combine for F1 score, but should be
    # a reasonable estimate.
    df["f1_errn95"] = harmonic_mean(df.precision_errn95, df.recall_errn95)


##
# Old scoring helpers (TODO: remove these)
def get_precision(df):
    return df.tp_count / (df.tp_count + df.fp_count)


def get_recall(df):
    return df.tp_count / (df.tp_count + df.fn_count)


def get_f1(df):
    return 2 * df.precision * df.recall / (df.precision + df.recall)


def _get_nested_vals(record, outer_key):
    return {f"{outer_key}.{key}": value for key, value in record.get(outer_key, {}).items()}


def _expand_runs(seq, nreps):
    """Repeat each element N times, consecutively.

    i.e. _expand_runs([1,2,3], 4) -> [1,1,1,1,2,2,2,2,3,3,3,3]
    """
    return np.tile(seq, (nreps, 1)).T.flatten()


class EdgeProbingExample(object):
    """Wrapper object to handle an edge probing example.

    Mostly exists for pretty-printing, but could be worth integrating
    into the data-handling code.
    """

    def __init__(self, record: Dict, label_vocab: List[str] = None, pred_thresh: float = 0.5):
        """Construct an example from a record.

        Record should be derived from the standard JSON format.
        """
        self._data = record
        self._label_vocab = label_vocab
        self._pred_thresh = pred_thresh

    @staticmethod
    def format_span(tokens, s, e, max_tok=None, render_fn=lambda tokens: " ".join(tokens)):
        selected_tokens = tokens[s:e]
        if max_tok is not None and len(selected_tokens) > max_tok:
            selected_tokens = selected_tokens[:max_tok] + ["..."]
        return '[{:2d},{:2d})\t"{:s}"'.format(s, e, render_fn(selected_tokens))

    def _fmt_preds(self, preds):
        buf = io.StringIO()
        for i, p in enumerate(preds["proba"]):
            if p < self._pred_thresh:
                continue
            buf.write("  {:5s}  ".format("" if buf.getvalue() else "pred:"))
            label = self._label_vocab[i] if self._label_vocab else str(i)
            buf.write(f"\t\t {label:s} ({p:.2f})\n")
        return buf.getvalue()

    def __str__(self):
        buf = io.StringIO()
        text = self._data["text"]
        tokens = text.split()
        buf.write("Text ({:d}): {:s}\n".format(len(tokens), text))

        for t in self._data["targets"]:
            buf.write("\n")
            buf.write("  span1: {}\n".format(self.format_span(tokens, *t["span1"])))
            if "span2" in t:
                buf.write("  span2: {}\n".format(self.format_span(tokens, *t["span2"])))
            labels = utils.wrap_singleton_string(t["label"])
            buf.write("  label: ({:d})\t\t {}\n".format(len(labels), ", ".join(labels)))
            # Show predictions, if present.
            if "preds" in t:
                buf.write(self._fmt_preds(t["preds"]))

        return buf.getvalue()

    def __repr__(self):
        return "EdgeProbingExample(" + repr(self._data) + ")"


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
            d = {"text": r["text"], "idx": idx}
            d.update(_get_nested_vals(r, "info"))
            d.update(_get_nested_vals(r, "preds"))
            ex_records.append(d)

            for t in r["targets"]:
                d = {"label": utils.wrap_singleton_string(t["label"]), "idx": idx}
                if "span1" in t:
                    d["span1"] = tuple(t["span1"])
                if "span2" in t:
                    d["span2"] = tuple(t["span2"])
                d.update(_get_nested_vals(t, "info"))
                d.update(_get_nested_vals(t, "preds"))
                tr_records.append(d)
        return ex_records, tr_records

    def _labels_to_ids(self, labels: List[str]) -> List[int]:
        return [self.vocab.get_token_index(l, namespace=self.label_namespace) for l in labels]

    def _get_num_labels(self) -> int:
        return self.vocab.get_vocab_size(namespace=self.label_namespace)

    def _label_ids_to_khot(self, label_ids: List[int]) -> np.ndarray:
        arr = np.zeros(self._get_num_labels(), dtype=np.int32)
        arr[label_ids] = 1
        return arr

    def _get_label(self, i: int) -> str:
        return self.vocab.get_token_from_index(i, namespace=self.label_namespace)

    def __init__(self, vocab: Vocabulary, records: Iterable[Dict], label_namespace=None):
        self.vocab = vocab
        self.label_namespace = label_namespace
        self.all_labels = [self._get_label(i) for i in range(self._get_num_labels())]

        ex_records, tr_records = self._split_and_flatten_records(records)
        self.example_df = pd.DataFrame.from_records(ex_records)
        self.example_df.set_index("idx", inplace=True, drop=False)
        self.target_df = pd.DataFrame.from_records(tr_records)

        # Apply indexing to labels
        self.target_df["label.ids"] = self.target_df["label"].map(self._labels_to_ids)
        # Convert labels to k-hot to align to predictions
        self.target_df["label.khot"] = self.target_df["label.ids"].map(self._label_ids_to_khot)

        # Placeholders, will compute later if requested.
        # Use non-underscore versions to access via propert getters.
        self._target_df_wide = None  # wide-form targets (expanded)
        self._target_df_long = None  # long-form targets (melted by label)

    def _make_wide_target_df(self):
        log.info("Generating wide-form target DataFrame. May be slow... ")
        # Expand labels to columns
        expanded_y_true = self.target_df["label.khot"].apply(pd.Series)
        expanded_y_true.columns = ["label.true." + l for l in self.all_labels]
        expanded_y_pred = self.target_df["preds.proba"].apply(pd.Series)
        expanded_y_pred.columns = ["preds.proba." + l for l in self.all_labels]
        wide_df = pd.concat([self.target_df, expanded_y_true, expanded_y_pred], axis="columns")
        DROP_COLS = ["preds.proba", "label", "label.ids", "label.khot"]
        wide_df.drop(labels=DROP_COLS, axis=1, inplace=True)
        log.info("Done!")
        return wide_df

    @property
    def target_df_wide(self):
        """Target df in wide form. Compute only if requested."""
        if self._target_df_wide is None:
            self._target_df_wide = self._make_wide_target_df()
        return self._target_df_wide

    def _make_long_target_df(self):
        df = self.target_df
        log.info("Generating long-form target DataFrame. May be slow... ")
        num_targets = len(df)
        # Index into self.example_df for text.
        ex_idxs = _expand_runs(df["idx"], len(self.all_labels))
        # Index into self.target_df for other metadata.
        idxs = _expand_runs(df.index, len(self.all_labels))
        # Repeat labels for each target.
        labels = np.tile(self.all_labels, num_targets)
        # Flatten lists using numpy - *much* faster than using Pandas.
        label_true = np.array(df["label.khot"].tolist(), dtype=np.int32).flatten()
        preds_proba = np.array(df["preds.proba"].tolist(), dtype=np.float32).flatten()
        assert len(label_true) == len(preds_proba)
        assert len(label_true) == len(labels)
        d = {
            "idx": idxs,
            "label": labels,
            "label.true": label_true,
            "preds.proba": preds_proba,
            "ex_idx": ex_idxs,
        }
        # Repeat some metadata fields if available.
        # Use these for stratified scoring.
        if "info.height" in df.columns:
            log.info("info.height field detected; copying to long-form " "DataFrame.")
            d["info.height"] = _expand_runs(df["info.height"], len(self.all_labels))
        if "span2" in df.columns:
            log.info("span2 detected; adding span_distance to long-form " "DataFrame.")

            def _get_midpoint(span):
                return (span[1] - 1 + span[0]) / 2.0

            def span_sep(a, b):
                ma = _get_midpoint(a)
                mb = _get_midpoint(b)
                if mb >= ma:  # b starts later
                    return max(0, b[0] - a[1])
                else:  # a starts later
                    return max(0, a[0] - b[1])

            span_distance = [span_sep(a, b) for a, b in zip(df["span1"], df["span2"])]
            d["span_distance"] = _expand_runs(span_distance, len(self.all_labels))
        # Reconstruct a DataFrame.
        long_df = pd.DataFrame(d)
        log.info("Done!")
        return long_df

    @property
    def target_df_long(self):
        """Target df in long form (one row per label, per target).
        Compute only if requested.
        """
        if self._target_df_long is None:
            self._target_df_long = self._make_long_target_df()
        return self._target_df_long

    @staticmethod
    def score_long_df(df: pd.DataFrame) -> Dict[str, float]:
        """Compute metrics for a single DataFrame of long-form predictions."""
        # Confusion matrix; can compute other metrics from this later.
        y_true = df["label.true"]
        y_pred = df["preds.proba"] >= 0.5
        C = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = C.ravel()
        record = dict()
        record["tn_count"] = tn
        record["fp_count"] = fp
        record["fn_count"] = fn
        record["tp_count"] = tp
        return record

    def score_by_label(self) -> pd.DataFrame:
        """Compute metrics for each label, and in the aggregate."""
        long_df = self.target_df_long
        gb = long_df.groupby(by=["label"])
        records = []
        for label, idxs in gb.groups.items():
            sub_df = long_df.loc[idxs]
            record = self.score_long_df(sub_df)
            record["label"] = label
            records.append(record)
        score_df = pd.DataFrame.from_records(records)
        ##
        # Compute macro average
        agg_map = {}
        for col in score_df.columns:
            # TODO(ian): ths _score_ entries don't actually do anything,
            # since when run the DataFrame only contains '_counts' columns.
            if col.endswith("_score"):
                agg_map[col] = "mean"
            elif col.endswith("_count"):
                agg_map[col] = "sum"
            elif col == "label":
                pass
            else:
                log.warning("Unsupported column '%s'", col)
        macro_avg = score_df.agg(agg_map)
        macro_avg["label"] = "_macro_avg_"
        score_df = score_df.append(macro_avg, ignore_index=True)

        ##
        # Compute micro average
        micro_avg = pd.Series(self.score_long_df(long_df))
        micro_avg["label"] = "_micro_avg_"
        score_df = score_df.append(micro_avg, ignore_index=True)

        ##
        # Compute stratified scores by special fields
        for field in ["info.height", "span_distance"]:
            if field not in long_df.columns:
                continue
            log.info(
                "Found special field '%s' with %d unique values.",
                field,
                len(long_df[field].unique()),
            )
            gb = long_df.groupby(by=[field])
            records = []
            for key, idxs in gb.groups.items():
                sub_df = long_df.loc[idxs]
                record = self.score_long_df(sub_df)
                record["label"] = "_{:s}_{:s}_".format(field, str(key))
                record["stratifier"] = field
                record["stratum_key"] = key
                records.append(record)
            score_df = score_df.append(
                pd.DataFrame.from_records(records), ignore_index=True, sort=False
            )

        ##
        # Move "label" column to the beginning.
        cols = list(score_df.columns)
        cols.insert(0, cols.pop(cols.index("label")))
        score_df = score_df.reindex(columns=cols)
        return score_df

    @classmethod
    def from_run(cls, run_dir: str, task_name: str, split_name: str):
        # Load vocabulary
        exp_dir = os.path.dirname(run_dir.rstrip("/"))
        vocab_path = os.path.join(exp_dir, "vocab")
        log.info("Loading vocabulary from %s" % vocab_path)
        vocab = Vocabulary.from_files(vocab_path)
        label_namespace = f"{task_name}_labels"

        # Load predictions
        preds_file = os.path.join(run_dir, f"{task_name}_{split_name}.json")
        log.info("Loading predictions from %s" % preds_file)
        return cls(vocab, utils.load_json_data(preds_file), label_namespace=label_namespace)


class Comparison(object):
    """Container class to represent a pair of experiments."""

    def __init__(self, base: Predictions, expt: Predictions, label_filter=lambda label: True):
        assert len(base.all_labels) == len(expt.all_labels)
        assert len(base.example_df) == len(expt.example_df)
        assert len(base.target_df) == len(expt.target_df)

        self.base = base
        self.expt = expt

        # Score & compare
        log.info("Scoring base run...")
        self.base_scores = base.score_by_label()
        self.base_scores["run"] = "base"
        log.info("Scoring expt run...")
        self.expt_scores = expt.score_by_label()
        self.expt_scores["run"] = "expt"
        log.info("Done scoring!")

        _mask = self.base_scores["label"].map(label_filter)
        self.base_scores = self.base_scores[_mask]
        _mask = self.expt_scores["label"].map(label_filter)
        self.expt_scores = self.expt_scores[_mask]

        self.long_scores = pd.concat([self.base_scores, self.expt_scores])
        # Wide-form scores for direct comparison
        df = pd.merge(
            self.base_scores,
            self.expt_scores,
            on=["label", "true_count"],
            suffixes=("_base", "_expt"),
        )
        df["abs_diff_f1"] = df["f1_score_expt"] - df["f1_score_base"]
        # Compute relative error reduction
        df["rel_diff_f1"] = df["abs_diff_f1"] / (1 - df["f1_score_base"])
        # Net diffs, computed from accuracy
        df["net_diffs"] = (
            np.abs(df["acc_score_expt"] - df["acc_score_base"]) * len(base.target_df)
        ).astype(np.int32)
        df["base_headroom"] = ((1 - df["acc_score_base"]) * len(base.target_df)).astype(np.int32)
        df["expt_headroom"] = ((1 - df["acc_score_expt"]) * len(expt.target_df)).astype(np.int32)
        self.wide_scores = df

    def plot_scores(
        self,
        task_name,
        metric="f1",
        sort_field="expt_headroom",
        sort_ascending=False,
        row_height=400,
        palette=None,
    ):
        import bokeh
        import bokeh.plotting as bp

        _SCORE_COL = f"{metric:s}_score"
        _ABS_DIFF_COL = f"abs_diff_{metric:s}"
        _REL_DIFF_COL = f"rel_diff_{metric:s}"

        # Long-form data, for paired bar chart.
        # TODO: switch to using wide-form for all?
        long_df = self.long_scores.copy()
        long_df["row_key"] = list(zip(long_df["label"], long_df["run"]))
        long_df["fmt_score"] = long_df[_SCORE_COL].map(lambda s: "{:.02f}".format(s))
        long_ds = bokeh.models.ColumnDataSource(data=long_df)

        # Wide-form data, for counts & comparison metrics.
        wide_df = self.wide_scores.copy()
        wide_df["_diff_label_offset"] = 10 * np.sign(wide_df[_ABS_DIFF_COL])
        wide_df["_count_diff_offset"] = -1.25 * wide_df["_diff_label_offset"]

        # Formatting label text
        wide_df["_net_diff_count_label_text"] = wide_df["net_diffs"].map(
            lambda s: "~ {:d} ex".format(s) if s else ""
        )
        wide_df["_abs_diff_label_text"] = wide_df[_ABS_DIFF_COL].map(lambda s: "{:.02f}".format(s))
        wide_df["_rel_diff_label_text"] = wide_df[_REL_DIFF_COL].map(
            lambda s: "({:.02f})".format(s) if s else ""
        )
        wide_ds = bokeh.models.ColumnDataSource(data=wide_df)

        # Prepare shared categorical axis
        runs = sorted(long_df["run"].unique())
        labels = wide_df.sort_values(by=sort_field, ascending=sort_ascending)["label"]
        categories = list(itertools.product(labels, runs))

        if not palette:
            #  palette = bokeh.palettes.Spectral6
            #  palette = bokeh.palettes.Category10[len(runs)]
            palette = bokeh.palettes.Category20[len(runs)]
            #  palette = bokeh.palettes.Set2[len(runs)]
        fill_cmap = bokeh.transform.factor_cmap("run", palette, runs)
        width = 35 * len(categories)
        tools = "xwheel_zoom,xwheel_pan,xpan,save,reset"

        # Top plot: score bars
        factor_range = bokeh.models.FactorRange(
            *categories, range_padding=0.5, range_padding_units="absolute"
        )
        p1 = bp.figure(
            title=f"Performance by label ({task_name})",
            x_range=factor_range,
            y_range=[0, 1],
            width=width,
            height=row_height,
            tools=tools,
        )
        p1.vbar(
            x="row_key",
            top=_SCORE_COL,
            width=0.95,
            fill_color=fill_cmap,
            line_color=None,
            source=long_ds,
        )
        label_kw = dict(
            text_align="right",
            text_baseline="middle",
            y_offset=-3,
            text_font_size="11pt",
            angle=90,
            angle_units="deg",
        )
        score_labels = bokeh.models.LabelSet(
            x="row_key", y=_SCORE_COL, text="fmt_score", source=long_ds, **label_kw
        )
        p1.add_layout(score_labels)
        p1.xaxis.major_label_orientation = 1
        p1.yaxis.bounds = (0, 1)

        # Second plot: absolute diffs
        p2 = bp.figure(
            title=f"Absolute and (Relative) diffs by label ({task_name})",
            x_range=p1.x_range,
            width=width,
            height=row_height,
            tools=tools,
        )
        p2.vbar(
            x="label",
            top="rel_diff_f1",
            width=1.90,
            fill_color="DarkRed",
            fill_alpha=0.20,
            line_color=None,
            source=wide_ds,
        )
        p2.vbar(
            x="label",
            top="abs_diff_f1",
            width=1.90,
            fill_color="DarkRed",
            line_color=None,
            source=wide_ds,
        )
        label_kw = dict(
            text_align="center",
            text_baseline="middle",
            y_offset="_diff_label_offset",
            source=wide_ds,
        )
        delta_labels = bokeh.models.LabelSet(
            x="label", y="abs_diff_f1", text="_abs_diff_label_text", **label_kw
        )
        p2.add_layout(delta_labels)
        rel_labels = bokeh.models.LabelSet(
            x="label", y="rel_diff_f1", text="_rel_diff_label_text", **label_kw
        )
        p2.add_layout(rel_labels)
        count_labels = bokeh.models.LabelSet(
            x="label",
            y=0,
            y_offset="_count_diff_offset",
            text="_net_diff_count_label_text",
            text_align="center",
            text_baseline="middle",
            text_font_style="italic",
            text_color="gray",
            text_font_size="10pt",
            source=wide_ds,
        )
        p2.add_layout(count_labels)

        p2.y_range.start = -1.10
        p2.y_range.end = 1.10
        p2.yaxis.bounds = (-1, 1)
        # Hacky: Hide category labels, not needed on this plot.
        p2.xaxis.major_label_text_color = None
        p2.xaxis.major_label_text_font_size = "0pt"
        p2.xaxis.major_tick_line_color = None

        # Bottom plot: count bars
        p3 = bp.figure(
            title=f"Counts by label ({task_name})",
            x_range=p1.x_range,
            width=width,
            height=row_height,
            tools=tools,
        )
        p3.vbar(
            x="label",
            top="true_count",
            width=1.90,
            fill_color="orange",
            line_color=None,
            source=wide_ds,
        )
        label_kw = dict(text_align="center", text_baseline="top", y_offset=-5)
        count_labels = bokeh.models.LabelSet(
            x="label", y="true_count", text="true_count", source=wide_ds, **label_kw
        )
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
            p1.xaxis.group_label_orientation = np.pi / 2
            p1.plot_height += 150
            # Middle plot: hide labels.
            p2.xaxis.group_text_color = None
            p2.xaxis.group_text_font_size = "0pt"
            # Bottom plot: rotate labels, add height.
            p3.xaxis.group_label_orientation = np.pi / 2
            p3.plot_height += 75

        # Create plot layout.
        plots = bokeh.layouts.gridplot(
            [p1, p2, p3], ncols=1, toolbar_location="left", merge_tools=True, sizing_mode="fixed"
        )
        header = bokeh.models.Div(text=f"<h1>{task_name} sorted by '{sort_field}'</h1>", width=600)
        return bokeh.layouts.column(header, plots)


class MultiComparison(object):
    """Similar to Comparison, but handles more than 2 experiments.

    Renders grouped bar plot and count bars, but not diff plot.
    """

    def __init__(self, runs_by_name: collections.OrderedDict, label_filter=lambda label: True):
        num_labels = {k: len(v.all_labels) for k, v in runs_by_name.items()}
        assert len(set(num_labels.values())) == 1
        num_examples = {k: len(v.example_df) for k, v in runs_by_name.items()}
        assert len(set(num_examples.values())) == 1
        num_targets = {k: len(v.target_df) for k, v in runs_by_name.items()}
        assert len(set(num_targets.values())) == 1

        self.runs_by_name = runs_by_name

        self.scores_by_name = collections.OrderedDict()
        for name, run in self.runs_by_name.items():
            log.info("Scoring run '%s'" % name)
            score_df = run.score_by_label()
            score_df["run"] = name
            _mask = score_df["label"].map(label_filter)
            score_df = score_df[_mask]
            self.scores_by_name[name] = score_df

        log.info("Done scoring!")

        self.long_scores = pd.concat(self.scores_by_name.values())

    def plot_scores(
        self,
        task_name,
        metric="f1",
        sort_field="expt_headroom",
        sort_run=None,
        sort_ascending=False,
        row_height=400,
        cmap=None,
    ):
        import bokeh
        import bokeh.plotting as bp

        _SCORE_COL = f"{metric:s}_score"

        # Long-form data, for grouped bar chart.
        long_df = self.long_scores.copy()
        long_df["row_key"] = list(zip(long_df["label"], long_df["run"]))
        long_df["fmt_score"] = long_df[_SCORE_COL].map(lambda s: "{:.02f}".format(s))
        long_ds = bokeh.models.ColumnDataSource(data=long_df)

        # Single scored run, for plotting counts.
        sort_run = sort_run or list(self.scores_by_name.keys())[0]
        wide_df = self.scores_by_name[sort_run]
        wide_ds = bokeh.models.ColumnDataSource(data=wide_df)

        # Prepare shared categorical axis
        #  runs = sorted(long_df['run'].unique())
        runs = list(self.scores_by_name.keys())
        labels = wide_df.sort_values(by=sort_field, ascending=sort_ascending)["label"]
        #  labels = sorted(long_df['label'].unique())
        categories = list(itertools.product(labels, runs))

        if cmap:
            palette = [cmap[name] for name in runs]
        else:
            #  palette = bokeh.palettes.Spectral6
            #  palette = bokeh.palettes.Category10[len(runs)]
            palette = bokeh.palettes.Category20[len(runs)]
            #  palette = bokeh.palettes.Set2[len(runs)]
        fill_cmap = bokeh.transform.factor_cmap("run", palette, runs)
        width = 30 * len(categories) + 10 * len(self.scores_by_name)
        tools = "xwheel_zoom,xwheel_pan,xpan,save,reset"

        # Top plot: score bars
        factor_range = bokeh.models.FactorRange(
            *categories, range_padding=0.5, range_padding_units="absolute"
        )
        p1 = bp.figure(
            title=f"Performance by label ({task_name})",
            x_range=factor_range,
            y_range=[0, 1],
            width=width,
            height=row_height,
            tools=tools,
        )
        p1.vbar(
            x="row_key",
            top=_SCORE_COL,
            width=0.95,
            fill_color=fill_cmap,
            line_color=None,
            source=long_ds,
        )
        label_kw = dict(
            text_align="right",
            text_baseline="middle",
            y_offset=-3,
            text_font_size="11pt",
            angle=90,
            angle_units="deg",
        )
        score_labels = bokeh.models.LabelSet(
            x="row_key", y=_SCORE_COL, text="fmt_score", source=long_ds, **label_kw
        )
        p1.add_layout(score_labels)
        p1.xaxis.major_label_orientation = 1
        p1.yaxis.bounds = (0, 1)

        # Middle plot: doesn't make sense for n > 2 experiments.

        # Bottom plot: count bars
        p3 = bp.figure(
            title=f"Counts by label ({task_name})",
            x_range=p1.x_range,
            width=width,
            height=row_height,
            tools=tools,
        )
        p3.vbar(
            x="label",
            top="true_count",
            width=1.90,
            fill_color="orange",
            line_color=None,
            source=wide_ds,
        )
        label_kw = dict(text_align="center", text_baseline="top", y_offset=-5)
        count_labels = bokeh.models.LabelSet(
            x="label", y="true_count", text="true_count", source=wide_ds, **label_kw
        )
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
            p1.xaxis.group_label_orientation = np.pi / 2
            p1.plot_height += 150
            # Bottom plot: rotate labels, add height.
            p3.xaxis.group_label_orientation = np.pi / 2
            p3.plot_height += 75

        # Create plot layout.
        plots = bokeh.layouts.gridplot(
            [p1, p3], ncols=1, toolbar_location="left", merge_tools=True, sizing_mode="fixed"
        )
        header_txt = f"{task_name}, sorted by '{sort_run}.{sort_field}'"
        header = bokeh.models.Div(text=f"<h1>{header_txt}</h1>", width=600)
        return bokeh.layouts.column(header, plots)
