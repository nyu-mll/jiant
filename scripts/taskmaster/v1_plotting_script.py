import argparse
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import scripts.taskmaster.get_results_row as get_results_row
import scripts.taskmaster.grab_analysis_data as grab_analysis_data
import scripts.taskmaster.calculate_correlations as calculate_correlations


GRAY_CMAP = LinearSegmentedColormap(
    "GRAY_CMAP",
    segmentdata={
        "red": [(0, 0.9, 0.9), (1, 0.9, 0.9)],
        "green": [(0, 0.9, 0.9), (1, 0.9, 0.9)],
        "blue": [(0, 0.9, 0.9), (1, 0.9, 0.9)],
    },
    N=256,
)


NAME_DICT = {
    "rte_TRG": "RTE",
    "boolq_TRG": "BoolQ",
    "commitbank_TRG": "CB",
    "copa_TRG": "COPA",
    "multirc_TRG": "MultiRC",
    "record_TRG": "ReCoRD",
    "wic_TRG": "WiC",
    "winograd-coreference_TRG": "WSC",
    "commonsenseqa_TRG": "CSenseQA",
    "cosmosqa_TRG": "CosmosQA",
    "edges-ner-ontonotes_PRB": "EP-NER",
    "edges-srl-ontonotes_PRB": "EP-SRL",
    "edges-coref-ontonotes_PRB": "EP-Coref",
    "edges-spr1_PRB": "EP-SPR1",
    "edges-spr2_PRB": "EP-SPR2",
    "edges-dpr_PRB": "EP-DPR",
    "edges-rel-semeval_PRB": "EP-Rel",
    "se-probing-word-content_PRB": "SE-WC",
    "se-probing-tree-depth_PRB": "SE-TreeDepth",
    "se-probing-top-constituents_PRB": "SE-TopConst",
    "se-probing-bigram-shift_PRB": "SE-BShift",
    "se-probing-past-present_PRB": "SE-Tense",
    "se-probing-subj-number_PRB": "SE-SubjNum",
    "se-probing-obj-number_PRB": "SE-ObjNum",
    "se-probing-odd-man-out_PRB": "SE-SOMO",
    "se-probing-coordination-inversion_PRB": "SE-CoordInv",
    "edges-pos-ontonotes_PRB": "EP-POS",
    "edges-nonterminal-ontonotes_PRB": "EP-Const",
    "edges-dep-ud-ewt_PRB": "EP-UD",
    "se-probing-sentence-length_PRB": "SE-SentLen",
    "acceptability-wh_PRB": "AJ-Wh",
    "acceptability-def_PRB": "AJ-Def",
    "acceptability-conj_PRB": "AJ-Coord",
    "acceptability-eos_PRB": "AJ-EOS",
    "cola_PRB": "AJ-CoLA",
    "SocialIQA": "SocialIQA",
    "ccg": "CCG",
    "commonsenseqa": "CSenseQA",
    "cosmosqa": "CosmosQA",
    "hellaswag": "HellaSwag",
    "mnli": "MNLI",
    "qamr": "QAMR",
    "qasrl": "QA-SRL",
    "qqp": "QQP",
    "scitail": "SciTail",
    "sst": "SST-2",
    "avg_targ": "Avg. Target",
}
TARG_NAME_LIST = [
    "commitbank_TRG",
    "copa_TRG",
    "winograd-coreference_TRG",
    "rte_TRG",
    "multirc_TRG",
    "wic_TRG",
    "boolq_TRG",
    "commonsenseqa_TRG",
    "cosmosqa_TRG",
    "record_TRG",
]
PROB_NAME_LIST = [
    "edges-pos-ontonotes_PRB",
    "edges-ner-ontonotes_PRB",
    "edges-srl-ontonotes_PRB",
    "edges-coref-ontonotes_PRB",
    "edges-nonterminal-ontonotes_PRB",
    "edges-spr1_PRB",
    "edges-spr2_PRB",
    "edges-dpr_PRB",
    "edges-rel-semeval_PRB",
    "edges-dep-ud-ewt_PRB",
    "se-probing-sentence-length_PRB",
    "se-probing-word-content_PRB",
    "se-probing-tree-depth_PRB",
    "se-probing-top-constituents_PRB",
    "se-probing-bigram-shift_PRB",
    "se-probing-past-present_PRB",
    "se-probing-subj-number_PRB",
    "se-probing-obj-number_PRB",
    "se-probing-odd-man-out_PRB",
    "se-probing-coordination-inversion_PRB",
    "cola_PRB",
    "acceptability-wh_PRB",
    "acceptability-def_PRB",
    "acceptability-conj_PRB",
    "acceptability-eos_PRB",
]
INT_TASK_NAMES = [
    "qamr",
    "commonsenseqa",
    "scitail",
    "cosmosqa",
    "SocialIQA",
    "ccg",
    "hellaswag",
    "qasrl",
    "sst",
    "qqp",
    "mnli",
]


def holm_bonferoni(p_value, alpha=0.05):
    arr = p_value.values.reshape(-1)
    k = (np.sort(arr) > (alpha / (len(arr) + 1 - np.arange(len(arr))))).argmax()
    return np.argsort(np.argsort(arr)).reshape(*p_value.shape) < k


def bonferoni(p_value, alpha=0.05):
    return p_value < (alpha / np.prod(p_value.shape))


def fmt_num(x):
    if np.isnan(x):
        return ""
    elif np.isclose(x, 1):
        return "1"
    elif x < 0:
        return "-" + f"{np.abs(x):.2f}".lstrip("0")
    else:
        return f"{x:.2f}".lstrip("0")


def plot_small_correlations(results_df, correl_type, plot_path):
    correl, p_value, count = calculate_correlations.compute_correlation_stats(
        results_df, correl_type=correl_type
    )
    correl_sig = correl.copy()
    plot_df = correl_sig.loc[TARG_NAME_LIST].loc[:, TARG_NAME_LIST + PROB_NAME_LIST]
    plot_df.index = plot_df.index.map(NAME_DICT.get)
    plot_df.columns = plot_df.columns.map(NAME_DICT.get)
    plot_df2 = plot_df.copy()
    plot_df[
        ~holm_bonferoni(
            p_value.loc[TARG_NAME_LIST].loc[:, TARG_NAME_LIST + PROB_NAME_LIST], alpha=0.05
        )
    ] = np.NaN
    labels = plot_df.applymap(fmt_num)

    fig = plt.figure(figsize=(20, 8))
    ax = fig.gca()
    ax = sns.heatmap(
        plot_df2,
        cmap=GRAY_CMAP,
        vmin=-1,
        vmax=1,
        annot=False,
        fmt="",
        linewidths=1,
        annot_kws={"size": 16, "fontweight": "bold"},
        cbar=False,
        ax=ax,
    )
    ax = sns.heatmap(
        plot_df,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        annot=labels,
        fmt="",
        linewidths=1,
        annot_kws={"size": 16, "fontweight": "bold"},
        cbar=False,
        ax=ax,
    )
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontweight="bold", fontsize=18)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontweight="bold", fontsize=18)
    plt.xticks(rotation=90)
    plt.ylim(10, 0)

    plt.axvline(10, color="black", lw=3, ymin=0.005, ymax=1.57, clip_on=False, linestyle="solid")
    plt.axvline(20, color="gray", lw=1.5, ymin=0.005, ymax=1.45, clip_on=False, linestyle="dashed")
    plt.axvline(30, color="gray", lw=1.5, ymin=0.005, ymax=1.45, clip_on=False, linestyle="dashed")

    ax.text(4, -4.7, "Target", fontsize=30, fontfamily="Courier New")
    ax.text(21, -4.7, "Probing", fontsize=30, fontfamily="Courier New")

    plt.gcf().subplots_adjust(top=0.65, bottom=0.05, left=0.08, right=0.99)
    plt.savefig(plot_path, dpi=200, transparent=True)


def plot_full_correlations(results_df, correl_type, plot_path):
    correl, p_value, count = calculate_correlations.compute_correlation_stats(
        results_df, correl_type=correl_type
    )
    plot_df = (
        correl.loc[TARG_NAME_LIST + PROB_NAME_LIST].loc[:, TARG_NAME_LIST + PROB_NAME_LIST].copy()
    )
    plot_df.index = plot_df.index.map(NAME_DICT.get)
    plot_df.columns = plot_df.columns.map(NAME_DICT.get)
    labels = plot_df.applymap(fmt_num)

    plt.figure(figsize=(16, 16))
    ax = sns.heatmap(
        plot_df,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        annot=labels,
        fmt="",
        linewidths=1,
        annot_kws={"size": 12, "fontweight": "bold"},
        cbar=False,
    )
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontweight="bold", fontsize=13)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontweight="bold", fontsize=13)
    plt.xticks(rotation=90)
    plt.ylim(35, 0)

    plt.axvline(10, lw=3, color="black", ymin=0.0, ymax=1.1, clip_on=False)
    plt.axvline(20, lw=1.5, color="gray", ymin=0.0, ymax=1.1, clip_on=False, linestyle="dashed")
    plt.axvline(30, lw=1.5, color="gray", ymin=0.0, ymax=1.1, clip_on=False, linestyle="dashed")
    plt.axvline(34, lw=1.5, color="gray", ymin=0.0, ymax=1.1, clip_on=False, linestyle="dashed")

    plt.axhline(10, lw=3, color="black", xmin=-0.1, xmax=1.0, clip_on=False)
    plt.axhline(20, lw=1.5, color="gray", xmin=-0.1, xmax=1.0, clip_on=False, linestyle="dashed")
    plt.axhline(30, lw=1.5, color="gray", xmin=-0.1, xmax=1.0, clip_on=False, linestyle="dashed")
    plt.axhline(34, lw=1.5, color="gray", xmin=-0.1, xmax=1.0, clip_on=False, linestyle="dashed")

    plt.gcf().subplots_adjust(top=0.915, bottom=0.02, left=0.09, right=0.99)
    plt.savefig(plot_path, dpi=200, transparent=True)


def plot_transfer_table(averaged, baseline_srs, plot_path):
    diff = (averaged - baseline_srs).copy()
    diff["avg_targ"] = diff.filter(like="TRG").mean(1)
    plot_df = diff.loc[INT_TASK_NAMES].loc[:, TARG_NAME_LIST + ["avg_targ"] + PROB_NAME_LIST].copy()
    plot_df.index = plot_df.index.map(NAME_DICT.get)
    plot_df.columns = plot_df.columns.map(NAME_DICT.get)
    plot_df = plot_df.T

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(1, 12)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[0, -1])
    ax = sns.heatmap(
        plot_df * 100,
        cmap="RdBu",
        vmin=-30,
        vmax=30,
        annot=True,
        fmt=".1f",
        linewidths=1,
        annot_kws={"size": 13, "fontweight": "bold"},
        cbar=False,
        ax=ax1,
    )
    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontweight="bold", fontsize=12.5)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontweight="bold", fontsize=13.5)
    ax.set_ylim(36, 0)
    ax.set_ylabel("")
    ax.set_xlabel("")

    baseline_plus_avg_dict = baseline_srs.to_dict()
    baseline_plus_avg_dict["avg_targ"] = baseline_srs.filter(like="TRG").mean()
    baseline_plus_avg = pd.Series(baseline_plus_avg_dict).loc[
        TARG_NAME_LIST + ["avg_targ"] + PROB_NAME_LIST
    ]

    sns.heatmap(
        pd.DataFrame({"Baseline\nPerformance": 100 * baseline_plus_avg}),
        cmap=GRAY_CMAP,
        vmin=-0.3,
        vmax=0.3,
        annot=True,
        fmt=".1f",
        linewidths=1,
        annot_kws={"size": 12.5, "fontweight": "bold"},
        cbar=False,
        ax=ax2,
    )
    ax2.tick_params(axis="both", which="both", length=0)
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontweight="bold", fontsize=11.5)
    ax2.set_ylim(36, 0)
    ax2.set_yticks([])

    ax.axhline(0 + 10, color="black", lw=2, xmin=-0.16, xmax=1, clip_on=False, linestyle="solid")
    ax.axhline(1 + 10, color="black", lw=3.5, xmin=-0.16, xmax=1, clip_on=False, linestyle="solid")
    ax.axhline(
        1 + 10 + 10, color="gray", lw=1.5, xmin=-0.135, xmax=1, clip_on=False, linestyle="dashed"
    )
    ax.axhline(
        1 + 10 + 20, color="gray", lw=1.5, xmin=-0.135, xmax=1, clip_on=False, linestyle="dashed"
    )

    ax2.axhline(0 + 10, color="black", lw=2, xmin=-0.2, xmax=1, clip_on=False, linestyle="solid")
    ax2.axhline(1 + 10, color="black", lw=3.5, xmin=-0.2, xmax=1, clip_on=False, linestyle="solid")
    ax2.axhline(
        1 + 10 + 10, color="gray", lw=1.5, xmin=-0.15, xmax=1, clip_on=False, linestyle="dashed"
    )
    ax2.axhline(
        1 + 10 + 20, color="gray", lw=1.5, xmin=-0.15, xmax=1, clip_on=False, linestyle="dashed"
    )

    plt.gcf().subplots_adjust(top=0.94, bottom=0.02, left=0.135, right=0.98)
    ax1.text(-1.9, 6, "Target", rotation=90, fontsize=16, fontfamily="Courier New")
    ax1.text(-1.9, 23, "Probing", rotation=90, fontsize=16, fontfamily="Courier New")
    plt.savefig(plot_path, dpi=200, transparent=True)


def get_jiant_path():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))


def plot_all(output_base_path):
    jiant_path = get_jiant_path()
    os.makedirs(output_base_path, exist_ok=True)

    # === Correlations === #
    all_data = grab_analysis_data.load_raw_data(
        os.path.join(jiant_path, "scripts/taskmaster/metadata/v1_data_files/runs")
    )
    results_df = grab_analysis_data.get_correlation_raw_data(all_data)
    results_df = results_df[[col for col in results_df.columns if not col.endswith("_MIX")]]
    results_df.to_csv(os.path.join(output_base_path, "correlation_raw_data.csv"))
    plot_small_correlations(
        results_df=results_df,
        correl_type="spearman",
        plot_path=os.path.join(output_base_path, "correl_small_spearman"),
    )
    plot_full_correlations(
        results_df=results_df,
        correl_type="spearman",
        plot_path=os.path.join(output_base_path, "correl_big_spearman"),
    )
    plot_full_correlations(
        results_df=results_df,
        correl_type="pearson",
        plot_path=os.path.join(output_base_path, "correl_big_pearson"),
    )

    # === Transfer === #
    data_df = results_df.reset_index()
    single_metadata = pd.read_csv(
        os.path.join(
            jiant_path, "scripts/taskmaster/metadata/v1_data_files/single_task_metadata.csv"
        )
    )
    averaged = data_df.groupby("int_task").mean()
    del averaged["run"]
    srs = get_results_row.get_result_series(
        get_results_row.read_tsv(
            os.path.join(jiant_path, "scripts/taskmaster/metadata/v1_data_files/base_results2.tsv")
        ),
        duplicate="last",
    )
    df = srs.reset_index()["index"].str.split("_", expand=True)
    df.columns = ["task_name", "run_num", "baseprobe"]
    df["vals"] = srs.values
    baseline_data = []
    baseline_performance_dict = single_metadata.set_index("task_jiant_name")[
        "single_performance_mean"
    ].to_dict()
    probing_baseline_performance_dict = df.groupby("task_name")["vals"].mean().to_dict()
    for col in averaged.columns:
        if "_TRG" in col:
            val = baseline_performance_dict[col.replace("_TRG", "")]
        elif "_PRB" in col:
            val = probing_baseline_performance_dict.get(col.replace("_PRB", ""), np.NaN)
        else:
            val = np.NaN
        baseline_data.append(val)
    baseline_srs = pd.Series(baseline_data, averaged.columns)
    plot_transfer_table(
        averaged=averaged,
        baseline_srs=baseline_srs,
        plot_path=os.path.join(output_base_path, "transfer.pdf"),
    )


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("output_base_path")
    args = parser.parse_args()
    plot_all(args.output_base_path)


if __name__ == "__main__":
    main()
