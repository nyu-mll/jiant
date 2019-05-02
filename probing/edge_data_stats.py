#!/usr/bin/env python

# Helper script to get stats on edge-probing data.
#
# Usage:
#  python edge_data_stats.py -i /path/to/edge/probing/data/*.json -o stats.tsv
#
# Will print dataset size, num targets, etc. to stdout, and optionally write
# stats to a TSV file if -o <file> is given.

import argparse
import collections
import json
import logging as log
import os
import sys

import pandas as pd
from tqdm import tqdm

from data import utils

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def analyze_file(fname: str):
    pd.options.display.float_format = "{:.2f}".format
    log.info("Analyzing file: %s", fname)
    record_iter = utils.load_json_data(fname)
    stats = utils.EdgeProbingDatasetStats()
    stats.compute(record_iter)
    log.info(stats.format(_name=fname))
    return stats.to_series(_name=fname)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="output", type=str, default="", help="Output file (TSV).")
    parser.add_argument("-i", dest="inputs", type=str, nargs="+", help="Input files.")
    args = parser.parse_args(args)

    all_stats = []
    for fname in args.inputs:
        all_stats.append(analyze_file(fname))
    df = pd.DataFrame(all_stats)
    df.set_index("_name", inplace=True)
    if args.output:
        log.info("Writing stats table to %s", args.output)
        df.to_csv(args.output, sep="\t")


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
