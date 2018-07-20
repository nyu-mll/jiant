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
from src import retokenize

def retokenize_record(record):
    """Retokenize edge probing examples. Modifies in-place.

    This can be slow, so recommended to use as a pre-processing step.
    See retokenize_edge_data.py.
    """
    text = record['text']
    moses_tokens = utils.TOKENIZER.tokenize(text)
    cleaned_moses_tokens = utils.unescape_moses(moses_tokens)
    ta = retokenize.TokenAligner(text, cleaned_moses_tokens)
    record['text'] = " ".join(moses_tokens)
    for target in record['targets']:
        if 'span1' in target:
            target['span1'] = list(map(int,
                                       ta.project_span(*target['span1'])))
        if 'span2' in target:
            target['span2'] = list(map(int,
                                       ta.project_span(*target['span2'])))
    return record

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
    main(sys.argv[1:])
    sys.exit(0)

