#!/usr/bin/env python

# Helper script to split constituent data into POS and nonterminal groups.
#
# TODO to integrate this into the OntoNotes processing script to generate in
# one shot.
#
# Usage:
#  python split_constituent_data.py /path/to/edge/probing/data/*.json

import copy
import json
import logging as log
import os
import sys

from tqdm import tqdm

from jiant.utils import utils

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def split_record(record):
    pos_record = copy.deepcopy(record)
    non_record = copy.deepcopy(record)

    pos_record["targets"] = [t for t in record["targets"] if t["info"]["height"] == 1]
    non_record["targets"] = [t for t in record["targets"] if t["info"]["height"] > 1]
    return (pos_record, non_record)


def split_file(fname):
    dirname, base = os.path.split(fname)

    pos_dir = os.path.join(dirname, "pos")
    os.makedirs(pos_dir, exist_ok=True)
    new_pos_name = os.path.join(pos_dir, base)

    non_dir = os.path.join(dirname, "nonterminal")
    os.makedirs(non_dir, exist_ok=True)
    new_non_name = os.path.join(non_dir, base)

    log.info("Processing file: %s", fname)
    record_iter = list(utils.load_json_data(fname))
    log.info("  saving to %s and %s", new_pos_name, new_non_name)
    pos_fd = open(new_pos_name, "w")
    non_fd = open(new_non_name, "w")
    for record in tqdm(record_iter):
        pos_record, non_record = split_record(record)
        pos_fd.write(json.dumps(pos_record))
        pos_fd.write("\n")
        non_fd.write(json.dumps(non_record))
        non_fd.write("\n")


def main(args):
    for fname in args:
        split_file(fname)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
