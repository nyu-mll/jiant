#!/usr/bin/env python

# Script to convert GAP coreference data (https://arxiv.org/pdf/1810.05201.pdf)
# to edge probing format.
#
# Download the data from
# https://github.com/google-research-datasets/gap-coreference
#
# Usage:
#   ./convert-gap.py \
#       -i /path/to/gap-coreference/<filename>.tsv \
#       -o /path/to/probing/data/gap/<filename>.json
#
# The GAP dataset provides targets as character offsets and text strings.
# We convert this to token indices by splitting on spaces, and finding the
# minimal span of tokens that includes the char span
# (offset + len(target_text)). Note that this may be a superstring of the
# actual target, since we don't attempt to separate punctuation from adjacent
# tokens.

import sys
import os
import json
import re
import collections
import argparse

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import utils
import numpy as np
import pandas as pd
from tqdm import tqdm

from nltk.tokenize.simple import SpaceTokenizer
SPACE_TOKENIZER = SpaceTokenizer()

from typing import Dict, Tuple, List, Iterable

# Based on
# https://github.com/google-research-datasets/gap-coreference/blob/master/constants.py
# Mapping of (lowercased) pronoun form to gender value. Note that reflexives
# are not included in GAP, so do not appear here.
PRONOUNS = {
    'she': "FEMININE",
    'her': "FEMININE",
    'hers': "FEMININE",
    'he': "MASCULINE",
    'his': "MASCULINE",
    'him': "MASCULINE",
}

def char_span_to_token_span(spans: List[Tuple[int, int]],
                            char_start: int, char_end: int) -> Tuple[int, int]:
    """ Map a character span to the minimal containing token span.
    Args:
        spans: a list of end-exclusive character spans for each token
        char_start: starting character offset
        char_end: ending character offset (exclusive)
    Returns:
        (start, end) token indices, end-exclusive
    """
    # first span ending after target start
    tok_s = min(i for i, s in enumerate(spans) if s[1] >= char_start)
    # last span starting before target end
    tok_e = max(i for i, s in enumerate(spans) if s[0] <= char_end)
    return (tok_s, tok_e + 1)  # end-exclusive

def row_to_record(row: pd.Series) -> Dict:
    """ Convert a TSV row into an edge probing record. """
    record = {}
    record['text'] = row['Text']
    record['info'] = {'id': row["ID"], 'url': row["URL"]}
    record['targets'] = []

    spans = tuple(SPACE_TOKENIZER.span_tokenize(row['Text']))
    def _get_span(prefix):
        """ Get the token span for a given prefix (Pronoun, A, B). """
        char_start = row[prefix + "-offset"]
        char_end = char_start + len(row[prefix])
        return char_span_to_token_span(spans, char_start, char_end)

    def _validate_span(token_span, target_text):
        """ Make sure the expected text is a substring of the returned span. """
        cs = spans[token_span[0]][0]
        ce = spans[token_span[1] - 1][1]
        covered_text = row['Text'][cs:ce]
        assert target_text in covered_text

    pronoun_span = _get_span("Pronoun")
    target = {}
    pronoun = row['Pronoun']
    target['gender'] = PRONOUNS[pronoun.lower()]
    target['span1'] = pronoun_span
    target['span2'] = _get_span('A')
    target['span3'] = _get_span('B')
    _validate_span(target['span2'], row['A'])
    _validate_span(target['span3'], row['B'])
    A_label = str(int(row['A' + "-coref"]))
    B_label =  str(int(row['B' + "-coref"]))
    if A_label == '1':
        target["label"] = "first"
    elif B_label == '1':
        target["label"] = "second"
    else:
        target["label"] = "neither"
    target['info'] = {}
    record['targets'].append(target)

    return record

def convert_file(fname: str, target_fname: str):
    df = pd.read_csv(fname, sep='\t', header=0)

    records = (row_to_record(row) for i, row in df.iterrows())
    records = tqdm(records, total=len(df))
    utils.write_file_and_print_stats(records, target_fname)

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', type=str, required=True,
                        help="Input .tsv file.")
    parser.add_argument('-o', dest='output', type=str, required=True,
                        help="Output .json file.")
    args = parser.parse_args(args)

    pd.options.display.float_format = '{:.2f}'.format
    log.info("Converting %s", args.input)
    convert_file(args.input, args.output)


if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)

