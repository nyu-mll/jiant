#!/usr/bin/env python

# Helper script to retokenize edge-probing data using a BPE model.
# Uses the tokenizer for the OpenAI transformer LM, as described in
# https://blog.openai.com/language-unsupervised/ and using the code from
# https://github.com/openai/finetune-transformer-lm
#
# Like retokenize_edge_data.py, this saves the result alongside the original
# files.
#
# Usage:
#  python retokenize_edge_data.py /path/to/edge/probing/data/*.json
#
# Requirements: this requires the `spacy` and `ftfy` packages for running the
# preprocessing for the OpenAI model. These should only be needed for this
# script - main.py shouldn't need to do any further preprocessing.

import sys
import os
import json
from tqdm import tqdm

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

from src.utils import utils
from src.utils import retokenize
from src.openai_transformer_lm import utils as openai_utils

def space_tokenize_with_eow(sentence):
    """Add </w> markers to ensure word-boundary alignment."""
    return [t + "</w>" for t in sentence.split()]

def retokenize_record(record):
    """Retokenize edge probing examples. Modifies in-place.

    This can be slow, so recommended to use as a pre-processing step.
    See retokenize_edge_data.py.
    """
    text = record['text']
    eow_tokens = space_tokenize_with_eow(text)
    bpe_tokens = openai_utils.tokenize(text)

    ta = retokenize.TokenAligner(eow_tokens, bpe_tokens)
    record['text'] = " ".join(bpe_tokens)
    for target in record['targets']:
        if 'span1' in target:
            target['span1'] = list(map(int,
                                       ta.project_span(*target['span1'])))
        if 'span2' in target:
            target['span2'] = list(map(int,
                                       ta.project_span(*target['span2'])))
    return record

def retokenize_file(fname):
    new_tokenizer_name = "OpenAI.BPE"
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

