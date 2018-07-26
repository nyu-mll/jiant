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

import numpy as np
import pandas as pd

from src import utils

from typing import Dict

def update_stats(record: Dict, stats: Dict):
    stats['count'] += 1
    tokens = record['text'].split()
    stats['token.count'] += len(tokens)
    stats['token.count2'] += len(tokens)**2  # for computing RMS
    
    # Target stats
    targets = record.get('targets', [])
    stats['targets.count'] += len(targets)
    for target in targets:
        labels = utils.wrap_singleton_string(target['label'])
        stats['targets.label.count'] += len(labels)
        span1 = target.get('span1', [-1,-1])
        stats['targets.span1.length'] += (max(span1) - min(span1))
        span2 = target.get('span2', [-1,-1])
        stats['targets.span2.length'] += (max(span2) - min(span2))

def print_stats(stats: Dict):
    s = pd.Series(dtype=object)
    s['count'] = stats['count']
    s['token.count'] = stats['token.count']
    s['token.mean_count'] = stats['token.count'] / stats['count']
    s['token.rms_count'] = np.sqrt(stats['token.count2'] / stats['count'])
    s['targets.count'] = stats['targets.count']
    s['targets.mean_count'] = stats['targets.count'] / stats['count']
    s['targets.label.count'] = stats['targets.label.count']
    s['targets.label.mean_count'] = stats['targets.label.count'] / stats['targets.count']
    s['targets.span1.mean_length'] = stats['targets.span1.length'] / stats['targets.count']
    s['targets.span2.mean_length'] = stats['targets.span2.length'] / stats['targets.count']
    log.info("Stats:\n%s\n", str(s))


def analyze_file(fname: str):
    pd.options.display.float_format = '{:.2f}'.format
    log.info("Analyzing file: %s", fname)
    record_iter = utils.load_json_data(fname)
    stats = collections.Counter()
    for record in record_iter:
        update_stats(record, stats)
    print_stats(stats)

def main(args):
    for fname in args:
        analyze_file(fname)

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)

