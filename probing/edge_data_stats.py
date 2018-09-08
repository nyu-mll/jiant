#!/usr/bin/env python

# Helper script to get stats on edge-probing data.
#
# Usage:
#  python edge_data_stats.py /path/to/edge/probing/data/*.json
#
# Will print dataset size, num targets, etc. to stdout.

import sys
import os
import json
import collections
from tqdm import tqdm

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import pandas as pd
from data import utils

def analyze_file(fname: str):
    pd.options.display.float_format = '{:.2f}'.format
    log.info("Analyzing file: %s", fname)
    record_iter = utils.load_json_data(fname)
    stats = utils.EdgeProbingDatasetStats()
    stats.compute(record_iter)
    log.info(stats.format())

def main(args):
    for fname in args:
        analyze_file(fname)

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)

