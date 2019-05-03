#!/usr/bin/env python

# Script to convert DPR data into edge probing format.
#
# Usage:
#   ./convert-dpr.py --src_dir <dpr_temp_dir> \
#       -o /path/to/probing/data/dpr

import argparse
import json
import logging as log
import os
import sys

import pandas as pd
from nltk.tokenize.moses import MosesTokenizer

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


TOKENIZER = MosesTokenizer()


def get_dpr_text(filename):
    text2examples = {}
    curr = {}
    with open(filename) as fd:
        for line in fd:
            line = line.strip()
            if not line:
                # store curr
                curr_text = curr["text"]
                if curr_text not in text2examples:
                    text2examples[curr_text] = []
                text2examples[curr_text].append(curr)
                # make new curr
                curr = {}
            else:
                # get id
                line = line.split(":")
                key = line[0].strip()
                val = " ".join(line[1:]).strip()
                curr[key] = val
    return text2examples


def convert_text_examples_to_json(text, example):
    # dict_keys(['provenance', 'index', 'text', 'hypothesis', 'entailed', 'partof'])
    # This assert makes sure that no text appears in train and test
    tokens = TOKENIZER.tokenize(text)
    split = set([ex["partof"] for ex in example])
    assert len(split) == 1
    obj = {
        "text": " ".join(tokens),
        "info": {"split": list(split)[0], "source": "recast-dpr"},
        "targets": [],
    }
    for ex in example:
        hyp = TOKENIZER.tokenize(ex["hypothesis"])
        assert len(tokens) <= len(hyp)
        found_diff_word = False
        for idx, pair in enumerate(zip(tokens, hyp)):
            if pair[0] != pair[1]:
                referent = ""
                found_diff_word = True
                distance = len(hyp) - len(tokens) + 1
                pro_noun = tokens[idx]
                found_referent = False
                for word_idx in range(idx + 1):
                    referent = hyp[idx : idx + distance]
                    if word_idx == 0:
                        referent[0] = referent[0][0].upper() + referent[0][1:]
                    if referent == tokens[word_idx : word_idx + distance]:
                        found_referent = True
                        target = {
                            "span1": [idx, idx + 1],
                            "span2": [word_idx, word_idx + distance],
                            "label": ex["entailed"],
                            "span1_text": pro_noun,
                            "span2_text": " ".join(tokens[word_idx : word_idx + distance]),
                        }
                        obj["targets"].append(target)
                        break
                break

    return obj


def convert_dpr(text2examples, output_dir: str):
    split_files = {
        k: open(os.path.join(output_dir, f"{k}.json"), "w") for k in ["train", "dev", "test"]
    }
    skip_counter = 0

    for text, example in text2examples.items():
        record = convert_text_examples_to_json(text, example)
        if not record.get("targets", []):
            skip_counter += 1
            continue
        # Write to file by split key.
        split = record["info"]["split"]
        split_files[split].write(json.dumps(record))
        split_files[split].write("\n")
    log.info("Skipped %d examples with no targets found.", skip_counter)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Path to inference_is_everything source data, as passed to get_dpr_data.sh",
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory, e.g. /path/to/edges/data/dpr",
    )
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    src_file = os.path.join(args.src_dir, "dpr_data.txt")
    text2examples = get_dpr_text(src_file)
    log.info("Processing DPR annotations...")
    convert_dpr(text2examples, output_dir=args.output_dir)
    log.info("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
