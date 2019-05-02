#!/usr/bin/env python

# Script to convert SPR2 data into edge probing format.
# Run get_spr2_data.sh <spr2_temp_dir> first to download the data.
#
# Usage:
#   ./convert-spr2.py --src_dir <spr2_temp_dir> \
#       -o /path/to/probing/data/spr2

import argparse
import json
import logging as log
import os
import sys

import conllu
import pandas as pd

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def load_ud_corpus(ud_source_dir: str):
    """
    Extracts the underlying UD corpus data that is stored in conllu format.
    Returns a dictionary where the keys are the split and the values are dictionaries
    where the keys are the sentenceId
    """
    data_path = os.path.join(ud_source_dir, "UD_English-EWT-r1.2")

    sent_id_to_text = {}
    for split in ["train", "dev", "test"]:
        split_path = os.path.join(data_path, f"en-ud-{split}.conllu")
        log.info("Loading UD data from %s", split_path)
        with open(split_path) as fd:
            data = "".join(line for line in fd)
        data = conllu.parse(data)
        sent_count = 0
        for sent in data:
            sent_id_to_text[(split, sent_count)] = " ".join([item["form"] for item in sent])
            sent_count += 1

    return sent_id_to_text


def convert_spr(sent_id_to_text, protoroles_source_dir: str, output_dir: str):
    sent_id2pred_arg_pairs = {}
    sent_id2targets = {}
    annotations_csv_path = os.path.join(protoroles_source_dir, "protoroles_eng_ud1.2_11082016.tsv")
    df = pd.read_csv(annotations_csv_path, sep="\t", header=0)
    for df_idx, row in df.iterrows():
        if row["Applicable"] == "no":
            continue
        id_pair = row["Sentence.ID"].split()
        assert len(id_pair) == 2
        split = id_pair[0].split(".con")[0].split("-")[-1]
        sent_id = id_pair[1]
        sent_text = sent_id_to_text[split, int(sent_id) - 1]
        if (split, sent_id) not in sent_id2targets:
            sent_id2targets[(split, sent_id)] = {
                "targets": [],
                "info": {
                    "source": "SPR2",
                    "sent-id": sent_id,
                    "split": split,
                    "grammatical": row["Sent.Grammatical"],
                },
            }
            sent_id2pred_arg_pairs[(split, sent_id)] = {}

        # span1 is predicate, span2 is argument
        span1 = (row["Pred.Token"], row["Pred.Token"] + 1)
        span2 = (row["Arg.Tokens.Begin"], row["Arg.Tokens.End"] + 1)
        if (span1, span2) not in sent_id2pred_arg_pairs[(split, sent_id)]:
            sent_id2pred_arg_pairs[(split, sent_id)][(span1, span2)] = {
                "span2": list(span2),
                "span1": list(span1),
                "label": {},
                "info": {
                    "span2_txt": row["Arg.Phrase"],
                    "span1_text": sent_text.split()[row["Pred.Token"]],
                    "is_pilot": row["Is.Pilot"],
                    "pred_lemma": row["Pred.Lemma"],
                },
            }

        _properties = sent_id2pred_arg_pairs[(split, sent_id)][(span1, span2)]["label"]
        if row["Property"] not in _properties:
            sent_id2pred_arg_pairs[(split, sent_id)][(span1, span2)]["label"][row["Property"]] = []
        sent_id2pred_arg_pairs[(split, sent_id)][(span1, span2)]["label"][row["Property"]].append(
            row["Response"]
        )

    outfiles = {}
    for key in sent_id2targets:
        val = sent_id2targets[key]
        val["text"] = sent_id_to_text[key[0], int(key[1]) - 1]
        val["info"]["split"] = key[0]
        val["info"]["sent_id"] = key[1]
        for span_pair in sent_id2pred_arg_pairs[key]:
            labels = sent_id2pred_arg_pairs[key][span_pair]["label"]
            sent_id2pred_arg_pairs[key][span_pair]["label"] = [
                key for key, val in labels.items() if sum(val) / float(len(val)) >= 4.0
            ]
            val["targets"].append(sent_id2pred_arg_pairs[key][span_pair])

        data_split = val["info"]["split"]
        if data_split not in outfiles:
            output_file = os.path.join(output_dir, f"edges.{data_split}.json")
            outfiles[data_split] = open(output_file, "w")
        json.dump(val, outfiles[data_split])
        outfiles[data_split].write("\n")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Path to source data (SPR and UD1.2), as passed " "to get_spr_data.sh",
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory, e.g. /path/to/edges/data/spr2",
    )
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    sent_id_to_text = load_ud_corpus(os.path.join(args.src_dir, "ud"))
    log.info("Processing proto-role annotations...")
    convert_spr(
        sent_id_to_text, os.path.join(args.src_dir, "protoroles"), output_dir=args.output_dir
    )
    log.info("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
