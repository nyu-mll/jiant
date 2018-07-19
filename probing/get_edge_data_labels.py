#!/usr/bin/env python

# Helper script to extract set of labels from edge probing data.
#
# Usage:
#  python get_edge_data_labels.py -o /path/to/edge/probing/data/labels.txt \
#      -i /path/to/edge/probing/data/*.json
#

import sys
import os
import argparse
import json
import collections
from tqdm import tqdm
from typing import Type

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

from src import utils

def retokenize_file(fname):
    new_tokenizer_name = utils.TOKENIZER.__class__.__name__
    new_name = fname + ".retokenized." + new_tokenizer_name
    log.info("Processing file: %s", fname)
    record_iter = list(utils.load_json_data(fname))
    log.info("  saving to %s", new_name)
    with open(new_name, 'w') as fd:
        for record in tqdm(record_iter):
            new_record = retokenize_record(record)
            fd.write(json.dumps(new_record))
            fd.write("\n")

def count_labels(fname: str) -> Type[collections.Counter]:
    label_ctr = collections.Counter()
    record_iter = utils.load_json_data(fname)
    for record in tqdm(record_iter):
        for target in record['targets']:
            label = target['label']
            if isinstance(label, str):
                label = [label]
            label_ctr.update(label)
    return label_ctr

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', dest='output', type=str, required=True,
                        help="Output file.")
    parser.add_argument('-i', dest='inputs', type=str, nargs="+",
                        help="Input files.")
    parser.add_argument('-s', dest='special_tokens', type=str,
                        nargs="*", default=["-"],
                        help="Additional special tokens to add at beginning "
                             "of vocab list.")
    args = parser.parse_args(args)

    label_ctr = collections.Counter()
    for fname in args.inputs:
        log.info("Counting labels in %s", fname)
        label_ctr.update(count_labels(fname))
    all_labels = args.special_tokens + sorted(label_ctr.keys())
    log.info("%d labels in total (%d special + %d found)",
             len(all_labels), len(args.special_tokens), len(label_ctr))
    with open(args.output, 'w') as fd:
        for label in all_labels:
            fd.write(label + "\n")

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)

