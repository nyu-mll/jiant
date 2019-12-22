# Reads a results.tsv file from the of new NLI data experiments and produces a
# set of LaTeX tables for use in the paper.

import csv
from collections import defaultdict
import re
from statistics import median, mean, stdev

main_columns = [
    "avg",
    "boolq-acc",
    "cb-f1",
    "cb-acc",
    "copa-acc",
    "multirc-f1a",
    "multirc-em",
    "record-f1",
    "record-em",
    "rte-acc",
    "wic-acc",
    "wsc-acc",
    "axb-mcc",
    "axg-gps",
    "axg-acc",
]
main_rows = [
    "None",
    "\\taska",
    "\\taskb",
    "\\taskc",
    "\\taskd",
    "\\taskf",
    "\\mnli 9k",
    "\\mnli Gov9k",
    "\\anli 9k",
    "\\mnli",
    "\\anli",
]

nli_columns = [
    "self",
    "mnli",
    "hans-acc",
    "glue-diagnostic_lex_sem",
    "glue-diagnostic_pr_ar_str",
    "glue-diagnostic_logic",
    "glue-diagnostic_knowledge",
    "glue-diagnostic_all_mcc",
]
nli_rows = [
    "\\taska",
    "\\taskb",
    "\\taskc",
    "\\taskd",
    "\\mnli 9k",
    "\\mnli Gov9k",
    "\\anli 9k",
    "\\mnli",
    "\\anli",
    "\\mnli~(two-class)",
    "\\mnli 9k (two-class)",
    "\\taskf",
]

ho_columns = ["self", "mnli"]
ho_rows = [
    "\\taska",
    "\\taskb",
    "\\taskc",
    "\\taskd",
    "\\mnli 9k",
    "\\mnli Gov9k",
    "\\anli 9k",
    "\\mnli",
    "\\anli",
    "\\mnli~(two-class)",
    "\\mnli 9k (two-class)",
    "\\taskf",
]


# This should be the (concatenated) results file produced by the experiments described in
# generate_script.py and 191123-paper-runs.sh.
with open("results.tsv") as in_file:
    tsv_reader = csv.reader(in_file, delimiter="\t")
    main_table = {
        "RoBERTa (large)": defaultdict(lambda: defaultdict(list)),
        "XLNet (large cased)": defaultdict(lambda: defaultdict(list)),
    }
    ho_table = {"RoBERTa (large)": defaultdict(lambda: defaultdict(list))}

    for row in tsv_reader:
        if len(row) < 2:
            continue
        results = row[1].split(",")
        groups = re.match("^([^-]*)-([^-]*)-([^-]*)-([^-]*)-([^-]*)", row[0]).groups()

        date = groups[0]
        category = groups[1]
        model = groups[2]
        training_data = groups[3]
        repeat = groups[4]

        if model == "roberta":
            model = "RoBERTa (large)"
        elif model == "xlnet":
            model = "XLNet (large cased)"
            if category == "hyp":
                # Omitting XLNet hyp-only results
                continue

        else:
            print("Can't find model spec: ", row)
            continue

        if training_data == "00":
            data_desc = "None"
        elif training_data in ["a9", "b9", "c9", "d9", "f9"]:
            data_desc = "\\task" + training_data[0]
        elif training_data == "g9":
            data_desc = "\\mnli Gov9k"
        elif training_data == "m9":
            data_desc = "\\mnli 9k"
        elif training_data == "ma":
            data_desc = "\\mnli"
        elif training_data == "x9":
            data_desc = "\\anli 9k"
        elif training_data == "xa":
            data_desc = "\\anli"
        else:
            print("Can't find data spec: ", row)
            continue

        results = row[1].split(", ")
        for result in results:
            result_parts = result.split(": ")
            if len(result_parts) != 2:
                print("Can't parse result: ", result_parts)
                continue
            result_name = result_parts[0]
            result_data = (repeat, float(result_parts[1]))

            if "avg" in result_name:
                continue

            if category == "hyp":
                if "mnli-two-ho" in result_name:
                    if training_data == "f9":
                        ho_table[model][data_desc]["mnli"].append(result_data)
                    elif training_data == "ma":
                        ho_table[model]["\\mnli~(two-class)"]["mnli"].append(result_data)
                        ho_table[model]["\\mnli~(two-class)"]["self"].append(result_data)
                    elif training_data == "m9":
                        ho_table[model]["\\mnli 9k (two-class)"]["mnli"].append(result_data)
                        ho_table[model]["\\mnli 9k (two-class)"]["self"].append(result_data)
                else:
                    if "mnli-ho" in result_name:
                        ho_table[model][data_desc]["mnli"].append(result_data)

                    if (
                        ("nli-a" in result_name and training_data == "a9")
                        or ("nli-b" in result_name and training_data == "b9")
                        or ("nli-c" in result_name and training_data == "c9")
                        or ("nli-d" in result_name and training_data == "d9")
                        or ("nli-f" in result_name and training_data == "f9")
                        or ("adversarial" in result_name and "x" in training_data)
                        or ("government" in result_name and "g" in training_data)
                        or ("mnli-ho_accuracy" in result_name and "m" in training_data)
                    ):
                        ho_table[model][data_desc]["self"].append(result_data)
            else:
                if "mnli-two_accuracy" in result_name:
                    if training_data == "f9":
                        main_table[model][data_desc]["mnli"].append(result_data)
                    elif training_data == "ma":
                        main_table[model]["\\mnli~(two-class)"]["mnli"].append(result_data)
                        main_table[model]["\\mnli~(two-class)"]["self"].append(result_data)
                    elif training_data == "m9":
                        main_table[model]["\\mnli 9k (two-class)"]["mnli"].append(result_data)
                        main_table[model]["\\mnli 9k (two-class)"]["self"].append(result_data)

                if "commitbank_accuracy" in result_name:
                    main_table[model][data_desc]["cb-acc"].append(result_data)
                elif "commitbank_f1" in result_name:
                    main_table[model][data_desc]["cb-f1"].append(result_data)
                elif "copa_accuracy" in result_name:
                    main_table[model][data_desc]["copa-acc"].append(result_data)
                elif "winograd-coreference_acc" in result_name:
                    main_table[model][data_desc]["wsc-acc"].append(result_data)
                elif "boolq_accuracy" in result_name:
                    main_table[model][data_desc]["boolq-acc"].append(result_data)
                elif "multirc_ans_f1" in result_name:
                    main_table[model][data_desc]["multirc-f1a"].append(result_data)
                elif "multirc_em" in result_name:
                    main_table[model][data_desc]["multirc-em"].append(result_data)
                elif "rte-superglue_accuracy" in result_name:
                    main_table[model][data_desc]["rte-acc"].append(result_data)
                elif "record_em" in result_name:
                    main_table[model][data_desc]["record-em"].append(result_data)
                elif "record_f1" in result_name:
                    main_table[model][data_desc]["record-f1"].append(result_data)
                elif "wic_accuracy" in result_name:
                    main_table[model][data_desc]["wic-acc"].append(result_data)
                elif "winogender-diagnostic_accuracy" in result_name:
                    main_table[model][data_desc]["axg-acc"].append(result_data)
                elif "winogender-diagnostic_gender_parity" in result_name:
                    main_table[model][data_desc]["axg-gps"].append(result_data)
                elif "broadcoverage-diagnostic_all_mcc" in result_name:
                    main_table[model][data_desc]["axb-mcc"].append(
                        result_data
                    )  # TODO: Handle finer-grained metrics.
                elif "nli-a_accuracy" in result_name:
                    main_table[model][data_desc]["self"].append(result_data)
                elif "nli-b_accuracy" in result_name:
                    main_table[model][data_desc]["self"].append(result_data)
                elif "nli-c_accuracy" in result_name:
                    main_table[model][data_desc]["self"].append(result_data)
                elif "nli-d_accuracy" in result_name:
                    main_table[model][data_desc]["self"].append(result_data)
                elif "nli-f_accuracy" in result_name:
                    main_table[model][data_desc]["self"].append(result_data)
                elif "mnli-government_accuracy" in result_name:
                    main_table[model][data_desc]["self"].append(result_data)
                elif "adversarial_nli_accuracy" in result_name:
                    main_table[model][data_desc]["self"].append(result_data)
                elif "hans_accuracy" in result_name:
                    main_table[model][data_desc]["hans-acc"].append(result_data)
                elif "mnli_accuracy" in result_name:
                    main_table[model][data_desc]["mnli"].append(result_data)
                elif "glue-diagnostic" in result_name:
                    main_table[model][data_desc][result_name].append(result_data)
                else:
                    pass
                    # print("Don't need result: (main) ", result, row)


def superglue_score(row):
    totals = defaultdict(float)

    for metric in ["boolq-acc", "copa-acc", "rte-acc", "wic-acc", "wsc-acc"]:
        if metric not in row:
            return "--"

        result_dict = dict(row[metric])

        for i in range(3):
            if str(i) in result_dict:
                totals[i] += result_dict[str(i)]
            else:
                return "--"

    for metric in ["cb-f1", "cb-acc", "multirc-f1a", "multirc-em", "record-f1", "record-em"]:
        if metric not in row:
            return "--"

        result_dict = dict(row[metric])

        for i in range(3):
            if str(i) in result_dict:
                totals[i] += result_dict[str(i)] * 0.5
            else:
                return "--"

    means = []
    for i in range(3):
        means.append(totals[i] / 9.0 * 100)

    total_mean = mean(means)
    std = stdev(means)

    return f"{total_mean:.1f} ($\\pm$ {std:.1f}) "


def median_of_three(values):
    result_dict = dict(values)
    result_float = median(list(result_dict.values())) * 100

    if len(result_dict) == 3:
        return f"{result_float:.1f}"
    else:
        return f"*{result_float:.1f}"  # Asterisk to mark missing data


print("\n\n\n")
print("\t".join(["Training Data"] + main_columns))
for model in main_table:
    print("\\midrule\n\\multicolumn{16}{c}{\\bf " + model + "}\\\\\n\\midrule")
    for data in main_rows:
        result_list = [data]
        for key in main_columns:
            if key in main_table[model][data]:
                result_list.append(median_of_three(main_table[model][data][key]))
            elif key == "avg":
                result_list.append(superglue_score(main_table[model][data]))
            else:
                result_list.append("--")
        print(" & ".join(result_list) + "\\\\")


print("\n\n\n")
print("\t".join(["Training Data"] + nli_columns))
for model in main_table:
    print("\\midrule\n\\multicolumn{4}{c}{\\bf " + model + "}\\\\\n\\midrule")
    for data in nli_rows:
        result_list = [data]
        for key in nli_columns:
            if key in main_table[model][data]:
                result_list.append(median_of_three(main_table[model][data][key]))
            else:
                result_list.append("--")
        print(" & ".join(result_list) + "\\\\")


print("\n\n\n")
print("\t".join(["Training Data"] + ho_columns))
for model in ho_table:
    print("\\midrule\n\\multicolumn{3}{c}{\\bf " + model + "}\\\\\n\\midrule")
    for data in ho_rows:
        result_list = [data]
        for key in ho_columns:
            if key in ho_table[model][data]:
                result_list.append(median_of_three(ho_table[model][data][key]))
            else:
                result_list.append("--")
        print(" & ".join(result_list) + "\\\\")

# TODO: Generate avg.
