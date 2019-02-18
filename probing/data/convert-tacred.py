#!/usr/bin/env python

# Script to convert TACRED data into edge probing format.
# Download the data from the LDC first and unpack with:
# tar -xvzf LDC2018T24.tgz
#
# Usage:
#   ./convert-tacred.py -i <tacred_dir>/data/json/*.json \
#       -o /path/to/probing/data/tacred
#
# This generates two sets of files:
# rel/*.json: relation classification, where span1 is the subject, span2 is the
# object, and the label is the relation class.
#
# entity/*.json: entity labeling, where span1 is a nominal mention and the
# label is the entity type.

import sys
import os
import json
import re
import argparse

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import utils
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Dict

def convert_record_rel(source_record: Dict) -> Dict:
    """Convert TACRED format to edge probing with relation targets."""
    record = {}
    record['text'] = " ".join(source_record['token'])
    record['info'] = dict(id=source_record['id'],
                          docid=source_record['docid'])
    # TACRED has a single relation triple for each input
    target = {}
    target['label'] = source_record['relation']
    target['span1'] = [source_record['subj_start'],
                       source_record['subj_end'] + 1]
    target['span2'] = [source_record['obj_start'],
                       source_record['obj_end'] + 1]
    target['info'] = dict(span1_type=source_record['subj_type'],
                          span2_type=source_record['obj_type'])
    record['targets'] = [target]
    return record

def convert_record_entity(source_record: Dict) -> Dict:
    """Convert TACRED format to edge probing with entity-type targets."""
    record = {}
    record['text'] = " ".join(source_record['token'])
    record['info'] = dict(id=source_record['id'],
                          docid=source_record['docid'])
    # Make a target for each subject, object.
    subj_target = {}
    subj_target['label'] = source_record['subj_type']
    subj_target['span1'] = [source_record['subj_start'],
                       source_record['subj_end'] + 1]
    obj_target = {}
    obj_target['label'] = source_record['obj_type']
    obj_target['span1'] = [source_record['obj_start'],
                       source_record['obj_end'] + 1]

    record['targets'] = [subj_target, obj_target]
    return record

def convert_with_stats(source_records, target_fname, convert_fn):
    converted_records = (convert_fn(r) for r in tqdm(source_records))
    stats = utils.EdgeProbingDatasetStats()
    converted_records = stats.passthrough(converted_records)
    utils.write_json_data(target_fname, converted_records)
    log.info("Wrote examples to %s", target_fname)
    log.info(stats.format())

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='inputs', type=str, nargs="+",
                        help="Input files (JSON) for TACRED splits.")
    parser.add_argument('-o', dest='output_dir', type=str, required=True,
                        help="Output directory.")
    args = parser.parse_args(args)

    os.makedirs(os.path.join(args.output_dir, "rel"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "entity"), exist_ok=True)

    pd.options.display.float_format = '{:.2f}'.format
    for fname in args.inputs:
        # Load mega-record (list of examples)
        with open(fname) as fd:
            records = json.load(fd)

        # Create relation labeling data.
        log.info("Converting %s", fname)
        target_fname = os.path.join(args.output_dir, "rel",
                                    os.path.basename(fname))
        convert_with_stats(records, target_fname, convert_record_rel)

        # Create relation labeling data.
        log.info("Converting %s", fname)
        target_fname = os.path.join(args.output_dir, "entity",
                                    os.path.basename(fname))
        convert_with_stats(records, target_fname, convert_record_entity)


if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)


