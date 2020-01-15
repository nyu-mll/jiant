import itertools
import glob
import os
import numpy as np
import pandas as pd
import tqdm

import get_results_row as get_results_row


INT_TASK_NAME_LIST = [
    "sst",
    "SocialIQA",
    "qqp",
    "mnli",
    "scitail",
    "qasrl",
    "qamr",
    "cosmosqa",
    "ccg",
    "hellaswag",
    "commonsenseqa",
]
TARGET_TASK_NAME_LIST = [
    "rte",
    "boolq",
    "commitbank",
    "copa",
    "multirc",
    "record",
    "wic",
    "winograd-coreference",
    "commonsenseqa",
    "cosmosqa",
]
PROBING_TASK_NAME_LIST = [
    "edges-ner-ontonotes",
    "edges-srl-ontonotes",
    "edges-coref-ontonotes",
    "edges-spr1",
    "edges-spr2",
    "edges-dpr",
    "edges-rel-semeval",
    "se-probing-word-content",
    "se-probing-tree-depth",
    "se-probing-top-constituents",
    "se-probing-bigram-shift",
    "se-probing-past-present",
    "se-probing-subj-number",
    "se-probing-obj-number",
    "se-probing-odd-man-out",
    "se-probing-coordination-inversion",
    "edges-pos-ontonotes",
    "edges-nonterminal-ontonotes",
    "edges-dep-ud-ewt",
    "se-probing-sentence-length",
    "acceptability-wh",
    "acceptability-def",
    "acceptability-conj",
    "acceptability-eos",
    "cola",
]
NAME_LIST_DICT = {
    "stilts": TARGET_TASK_NAME_LIST,
    "probing": PROBING_TASK_NAME_LIST,
    "mixing": PROBING_TASK_NAME_LIST,
}
RUN_NUM_LIST = [1, 2, 3]


def strip_run_suffix(s):
    if "_run" in s:
        return s.split("_run")[0]
    elif "_mixrun" in s:
        return s.split("_mixrun")[0]
    else:
        return s


def add_index_suffix(srs, suffix):
    srs = srs.copy()
    srs.index = srs.index.map(lambda s: s + suffix)
    return srs


EMPTY_TARGET = pd.Series(np.NaN, index=TARGET_TASK_NAME_LIST)
EMPTY_PROBING = pd.Series(np.NaN, index=PROBING_TASK_NAME_LIST)


def load_raw_data(base_path):
    path_ls = glob.glob(os.path.join(base_path, "*/*/*.tsv"))
    all_data = {"stilts": {}, "probing": {}, "mixing": {}}
    for path in tqdm.tqdm(path_ls):
        tokens = path.split(".")[0].split("/")
        run_num = int(tokens[-3].replace("run", ""))
        phase = tokens[-2]
        int_task = tokens[-1]

        srs = get_results_row.get_result_series(get_results_row.read_tsv(path), duplicate="last")
        srs.index = srs.index.map(lambda k: k.replace("rte-superglue", "rte"))
        if "_run" in srs.index[0] or "_mixrun" in srs.index[0]:
            # v2
            # Only keep the correct run nums
            selected = srs[srs.index.map(lambda k: "run{}".format(run_num) in k)].copy()
            if phase == "stilts":
                selected = selected[selected.index.map(lambda k: "_run" in k)].copy()
            elif phase == "probing":
                selected = selected[selected.index.map(lambda k: "_run" in k)].copy()
            elif phase == "mixing":
                selected = selected[selected.index.map(lambda k: "_mixrun" in k)].copy()
            selected.index = selected.index.map(strip_run_suffix)
            selected = selected.reindex(NAME_LIST_DICT[phase])
            if len(selected) == 0:
                print("EMPTY:", path)
        else:
            # specific fix:
            if int_task == "commonsenseqa" and run_num == 1:
                srs = srs.loc[~srs.index.duplicated(keep="last")]

            # v1
            # No filter by run nums
            selected = srs.reindex(NAME_LIST_DICT[phase])
        all_data[phase][(int_task, run_num)] = selected
    return all_data


def get_correlation_raw_data(all_data):
    results_rows = []
    results_index = list(itertools.product(RUN_NUM_LIST, INT_TASK_NAME_LIST))
    for run_num, int_task in results_index:
        key = (int_task, run_num)
        target_data = add_index_suffix(all_data["stilts"].get(key, EMPTY_TARGET), "_TRG")
        probing_data = add_index_suffix(all_data["probing"].get(key, EMPTY_PROBING), "_PRB")
        mixing_data = add_index_suffix(all_data["mixing"].get(key, EMPTY_PROBING), "_MIX")
        row = pd.concat([target_data, probing_data, mixing_data])
        results_rows.append(row)
    results_df = pd.DataFrame(results_rows, index=pd.MultiIndex.from_tuples(results_index))
    results_df.index.names = ["run", "int_task"]
    return results_df


def get_regression_raw_data(all_data, single_metadata, double_metadata):
    single_metadata["log2_num_examples"] = np.log2(single_metadata["num_examples"])
    single_metadata = single_metadata.set_index("task_jiant_name")[
        ["single_performance_mean", "log2_num_examples", "vocab_size", "average_input_length"]
    ]
    double_metadata = double_metadata.set_index(["int_task_name", "target_task_name"])[
        ["vocab_overlap", "task_api_match"]
    ]

    # Form Y
    targ_srs = pd.DataFrame(all_data["stilts"]).unstack()
    targ_srs.index = targ_srs.index.swaplevel(0, 1)
    full_index = list(itertools.product(RUN_NUM_LIST, INT_TASK_NAME_LIST, TARGET_TASK_NAME_LIST))
    targ_srs = targ_srs.reindex(full_index)

    # Form X
    x_rows = []
    for run_num, int_task, targ_task in targ_srs.index.tolist():
        key = (int_task, run_num)
        probing_data = add_index_suffix(all_data["probing"].get(key, EMPTY_PROBING), "_PRB")
        targ_metadata = add_index_suffix(single_metadata.loc[targ_task], "_META_TRG")
        int_metadata = add_index_suffix(single_metadata.loc[int_task], "_META_INT")
        targ_int_metadata = add_index_suffix(double_metadata.loc[int_task, targ_task], "_META_X")
        row = pd.concat([probing_data, mixing_data, targ_metadata, int_metadata, targ_int_metadata])
        x_rows.append(row)

    x_df = pd.DataFrame(x_rows)
    x_df.index = targ_srs.index
    full_df = x_df.copy()
    full_df.insert(0, "targ", targ_srs)
    full_df.index.names = ["run_num", "int_task", "targ_task"]
    return full_df
