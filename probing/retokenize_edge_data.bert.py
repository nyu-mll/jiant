#!/usr/bin/env python

# Helper script to retokenize edge-probing data for the BERT model.
#
# Uses the BERT tokenization as described in https://arxiv.org/abs/1810.04805,
# and using the code from
# https://github.com/huggingface/pytorch-pretrained-BERT#usage
#
# Like retokenize_edge_data.py, this saves the result alongside the original
# files.
#
# Usage:
#  python retokenize_edge_data.bert.py --model bert-base-uncased \
#      /path/to/edge/probing/data/*.json
#
# Requirements: this requires the 'pytorch_pretrained_bert' package for
# loading the BERT models and running their tokenizer.

import sys
import os
import json
import re
import argparse
from tqdm import tqdm

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

from src.utils import utils
from src.utils import retokenize

from pytorch_pretrained_bert import BertTokenizer

def process_wordpiece_for_alignment(t):
    """Add <w> markers to ensure word-boundary alignment."""
    if t.startswith("##"):
        return re.sub(r"^##", "", t)
    else:
        return "<w>" + t

def space_tokenize_with_bow(sentence):
    """Add <w> markers to ensure word-boundary alignment."""
    return ["<w>" + t for t in sentence.split()]

def retokenize_record(record, bert_tokenizer_fn, do_lower_case: bool):
    """Retokenize edge probing examples. Modifies in-place.

    This can be slow, so recommended to use as a pre-processing step.
    See retokenize_edge_data.py.
    """
    text = record['text']
    # If using lowercase, do this for the source tokens for better matching.
    bow_tokens = space_tokenize_with_bow(text.lower() if do_lower_case else
                                         text)
    wpm_tokens = bert_tokenizer_fn(text)

    # Align using <w> markers for stability w.r.t. word boundaries.
    modified_wpm_tokens = list(map(process_wordpiece_for_alignment,
                                   wpm_tokens))
    ta = retokenize.TokenAligner(bow_tokens, modified_wpm_tokens)

    record['text'] = " ".join(wpm_tokens)
    for target in record['targets']:
        if 'span1' in target:
            target['span1'] = list(map(int,
                                       ta.project_span(*target['span1'])))
        if 'span2' in target:
            target['span2'] = list(map(int,
                                       ta.project_span(*target['span2'])))
    return record

def retokenize_file(fname, new_tokenizer_name, **record_kw):
    new_name = fname + ".retokenized." + new_tokenizer_name
    log.info("Processing file: %s", fname)
    record_iter = list(utils.load_json_data(fname))
    log.info("  saving to %s", new_name)
    with open(new_name, 'w') as fd:
        for record in tqdm(record_iter):
            new_record = retokenize_record(record, **record_kw)
            fd.write(json.dumps(new_record))
            fd.write("\n")

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help="Name of the BERT model, as passed to"
                        "BertTokenizer.from_pretrained(), e.g."
                        "'bert-base-uncased'.")
    parser.add_argument('inputs', type=str, nargs="+",
                        help="Input files.")
    args = parser.parse_args(args)

    # Load tokenizer
    do_lower_case = args.model.endswith('uncased')
    log.info("do_lower_case=%s", str(do_lower_case))
    tokenizer = BertTokenizer.from_pretrained(args.model,
                                              do_lower_case=do_lower_case)

    for fname in args.inputs:
        retokenize_file(fname, args.model,
                        bert_tokenizer_fn=tokenizer.tokenize,
                        do_lower_case=do_lower_case)

if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)

