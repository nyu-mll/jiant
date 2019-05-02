#!/usr/bin/env python

# Script to convert SPR1 data from Rachel Rudinger's JSON format used for
# (https://arxiv.org/pdf/1804.07976.pdf) into edge probing format.
#
# Note that reconstructing SPR1 from the raw data available on decomp.net is
# considerably more difficult. TODO to check in a full pipeline of scripts to
# join against PTB and PropBank annotations.
#
# Usage:
#    ./convert-spr1-rudinger.py -i /path/to/spr1/*.json \
#        -o /path/to/probing/data/spr1/
#
# This will print a bunch of stats for each file, which you can sanity-check
# against the published size of the dataset.
#
# Input should be JSON with a single record per line, with a format similar to:
#  {'tokens': ['William', 'Craig', ',', 'an', 'independent', 'record',
#              'promoter', ',', 'pleaded', 'guilty', 'to', 'payola', 'and',
#              'criminal', 'tax', 'charges', ',', 'according', 'to', 'a',
#              'statement', 'issued', 'by', 'Gary', 'Feess', ',', 'the',
#              'U.S.', 'attorney', 'here', '.'],
#   'pb': [],
#   'wsd': [],
#   'split': 'dev',
#   'fn': [],
#   'sent_id': '1940_1',
#   'spr1': [
#       {'alignment_src': 'alignment_loss_00',
#        'arg_idx': 11,
#        'pred_idx': 8,
#        'responses': [{'spr_property': 'awareness', 'response': 1.0},
#                      {'spr_property': 'change_of_location', 'response': 1.0},
#                      {'spr_property': 'change_of_state', 'response': 5.0},
#                      ...
#                      {'spr_property': 'created', 'response': 1.0},
#                      {'spr_property': 'destroyed', 'response': 1.0}]
#       },
#       {'alignment_src': 'alignment_loss_00',
#        'arg_idx': 1,
#        'pred_idx': 8,
#        'responses': [{'spr_property': 'awareness', 'response': 5.0},
#                      {'spr_property': 'change_of_location', 'response': 3.0},
#                      ...
#                      {'spr_property': 'change_of_state', 'response': 4.0},]
#       }
#       ...
#    ]
#  }

import argparse
import collections
import json
import logging as log
import os
import sys
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def binarize_labels(responses):
    scores_by_property = collections.defaultdict(lambda: [])
    for response in responses:
        scores_by_property[response["spr_property"]] = response["response"]
    avg_scores = {k: np.mean(v) for k, v in scores_by_property.items()}
    bin_scores = {k: (v > 3.0) for k, v in avg_scores.items()}
    pos_labels = [k for k, v in bin_scores.items() if v]
    return sorted(pos_labels)


def convert_record(source_record):
    record = {}
    record["text"] = " ".join(source_record["tokens"])
    record["info"] = dict(split=source_record["split"], sent_id=source_record["sent_id"])
    targets = []
    for source_target in source_record["spr1"]:
        p = source_target["pred_idx"]  # token index
        a = source_target["arg_idx"]  # token index
        labels = binarize_labels(source_target["responses"])
        targets.append(dict(span1=[p, p + 1], span2=[a, a + 1], label=labels))
    record["targets"] = targets
    return record


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", dest="inputs", type=str, nargs="+", help="Input files (JSON) for SPR1 splits."
    )
    parser.add_argument("-o", dest="output_dir", type=str, required=True, help="Output directory.")
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    pd.options.display.float_format = "{:.2f}".format
    for fname in args.inputs:
        log.info("Converting %s", fname)
        source_records = list(utils.load_json_data(fname))
        converted_records = (convert_record(r) for r in tqdm(source_records))
        stats = utils.EdgeProbingDatasetStats()
        converted_records = stats.passthrough(converted_records)
        target_fname = os.path.join(args.output_dir, os.path.basename(fname))
        utils.write_json_data(target_fname, converted_records)
        log.info("Wrote examples to %s", target_fname)
        log.info(stats.format())


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
