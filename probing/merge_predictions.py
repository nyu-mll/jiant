#!/usr/bin/env python

# Script to merge predictions from a series of runs on the same task.
# Use this to collapse the JSON prediction files into a single one,
# where targets.preds.proba will go from [num_labels] to [num_runs, num_labels]
# in the output, and all other fields are expected to be the same across runs.
#
# Usage:
#  python merge_predictions.py \
#      -i /path/to/experiments/<prefix>-*-<task>/run/*_<split>.json \
#      -o /path/to/experiments/<prefix>-<task>_<split>.merged.json
#

import sys
import os
import re
import json
import argparse
import glob
import copy
from tqdm import tqdm

import logging as log

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)

from data import utils
import pandas as pd

from typing import List, Tuple, Iterable, Dict


def merge_records(records):
    ret = copy.deepcopy(records[0])
    for target in ret["targets"]:
        # Make a list we can extend
        target["preds"]["proba"] = []

    for r in records:
        assert r["text"] == ret["text"]
        assert len(r["targets"]) == len(ret["targets"])
        for i, target in enumerate(ret["targets"]):
            assert r["targets"][i]["span1"] == target["span1"]
            assert r["targets"][i].get("span2", None) == target.get("span2", None)
            assert r["targets"][i]["label"] == target["label"]
            target["preds"]["proba"].append(r["targets"][i]["preds"]["proba"])

    return ret


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="inputs", type=str, nargs="+", help="Input files (json).")
    parser.add_argument("-o", dest="output", type=str, required=True, help="Output file (json).")
    args = parser.parse_args(args)

    # Sort inputs for stability.
    preds_files = sorted(args.inputs)
    log.info(f"Found {len(preds_files)} inputs:")
    for fname in preds_files:
        log.info("  " + fname)

    # Read the first file to count records.
    num_records = sum(1 for line in open(preds_files[0]))
    log.info(f"Found {num_records} lines in first file.")

    # Make parallel iterators for each file.
    record_iters = [utils.load_json_data(fname) for fname in preds_files]

    # Merge records and write to file.
    merge_iter = map(merge_records, zip(*record_iters))
    merge_iter = tqdm(merge_iter, total=num_records)
    pd.options.display.float_format = "{:.2f}".format
    utils.write_file_and_print_stats(merge_iter, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
