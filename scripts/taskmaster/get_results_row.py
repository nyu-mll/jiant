import argparse
import numpy as np
import pandas as pd

TARGET_NAME_LIST = [
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


MIXING_NAME_LIST = [
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
    "cola",
]

PROBING_NAME_LIST = [
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


def read_file_lines(path, mode="r", encoding="utf-8", strip_lines=False, **kwargs):
    with open(path, mode=mode, encoding=encoding, **kwargs) as f:
        lines = f.readlines()
    if strip_lines:
        return [line.strip() for line in strip_lines]
    else:
        return lines


def read_tsv(path):
    result_lines = read_file_lines(path)
    results = []
    for line in result_lines:
        task_name, metrics = line.strip().split("\t")
        single_result = {"task_name": task_name}
        for raw_metrics in metrics.split(", "):
            metric_name, metric_value = raw_metrics.split(": ")
            single_result[metric_name] = float(metric_value)
        results.append(single_result)
    return results


def get_result_series(full_results, duplicate="raise"):
    srs_dict = {}
    for single_result in full_results:
        if single_result["task_name"] in srs_dict:
            if duplicate == "raise":
                raise RuntimeError(single_result["task_name"])
            elif duplicate == "first":
                continue
            elif duplicate == "last":
                pass
            else:
                raise KeyError(duplicate)
        srs_dict[single_result["task_name"]] = single_result["macro_avg"]
    return pd.Series(srs_dict)


def format_for_gsheet(result_srs, name_list):
    tokens = [
        "{:.5f}".format(x) if not np.isnan(x) else "-" for x in result_srs.reindex(name_list).values
    ]
    return ",".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("experiment")
    parser.add_argument("--resolve_duplicate", default="last")
    args = parser.parse_args()

    if args.experiment.lower() == "stilts":
        name_list = TARGET_NAME_LIST
    elif args.experiment.lower() == "probing":
        name_list = PROBING_NAME_LIST
    elif args.experiment.lower() == "mixing":
        name_list = MIXING_NAME_LIST
    else:
        raise KeyError(args.experiment)

    result_srs = get_result_series(read_tsv(args.path), duplicate=args.resolve_duplicate)
    print(format_for_gsheet(result_srs, name_list))


if __name__ == "__main__":
    main()
