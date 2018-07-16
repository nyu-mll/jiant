#!/usr/bin/env python

# Helper script to retokenize edge-probing data.
# Uses the tokenizer in utils.TOKENIZER and saves result alongside original
# files.
#
# Usage:
#  python retokenize_edge_data.py /path/to/edge/probing/data/*.json
#
# Speed: takes around 2.5 minutes to process 90000 sentences on a single core.

import sys
import os
import json
from tqdm import tqdm

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

from src import utils
from src.edge_probing import retokenize_record

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

def main(args):
    for fname in args:
        retokenize_file(fname)

if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except BaseException:
        # Make sure we log the trace for any crashes before exiting.
        log.exception("Fatal error in main():")
        sys.exit(1)
    sys.exit(0)

