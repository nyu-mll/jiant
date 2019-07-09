#!/usr/bin/env python

"""
This file will preprocess the SuperGLUE Winograd Schema Challenge data, aligning the span indices
to the tokenizaer of choice, and saving as a JSON file.

An example of the span index transformation is below:
[Mr., Porter, is, nice] with span indices [0, 2] -> [Mr, ., Por, ter, is, nice ]
with span indices [0, 3].

Usage:
    Run the below command from the root directory
    python -m scripts.winograd.preprocess_winograd
    -t {tokenizer_name} --data_dir {path/to/directory}

The input file should be in jsonl form, with text and tags columns. The output will be
in JSON form. See realign_spans for more details.

"""

from typing import Tuple, List, Text
from jiant.utils import tokenizers
from jiant.utils import retokenize
from jiant.utils import utils
import argparse
import functools
import json
import os
import re
import sys
import multiprocessing
from tqdm import tqdm
import pandas as pd
import logging as log

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def realign_spans(record, tokenizer_name):
    """
    Builds the indices alignment while also tokenizing the input
    piece by piece.

    Parameters
    -----------------------
        record: dict with the below fields
            text: str
            targets: list of dictionaries
                label: bool
                span1_index: int, start index of first span
                span1_text: str, text of first span
                span2_index: int, start index of second span
                span2_text: str, text of second span
        tokenizer_name: str

    Returns
    ------------------------
        record: dict with the below fields:
            text: str in tokenized form
            targets: dictionary with the below fields
                -label: bool
                -span_1: (int, int) of token indices
                -span1_text: str, the string
                -span2: (int, int) of token indices
                -span2_text: str, the string
    """

    # find span indices and text
    text = record["text"].split()
    span1 = record["targets"][0]["span1_index"]
    span1_text = record["targets"][0]["span1_text"]
    span2 = record["targets"][0]["span2_index"]
    span2_text = record["targets"][0]["span2_text"]

    # construct end spans given span text space-tokenized length
    span1 = [span1, span1 + len(span1_text.strip().split())]
    span2 = [span2, span2 + len(span2_text.strip().split())]
    indices = [span1, span2]

    sorted_indices = sorted(indices, key=lambda x: x[0])
    current_tokenization = []
    span_mapping = {}

    # align first span to tokenized text
    aligner_fn = retokenize.get_aligner_fn(tokenizer_name)
    _, new_tokens = aligner_fn(" ".join(text[: sorted_indices[0][0]]))
    current_tokenization.extend(new_tokens)
    new_span1start = len(current_tokenization)
    _, span_tokens = aligner_fn(" ".join(text[sorted_indices[0][0] : sorted_indices[0][1]]))
    current_tokenization.extend(span_tokens)
    new_span1end = len(current_tokenization)
    span_mapping[sorted_indices[0][0]] = [new_span1start, new_span1end]

    # re-indexing second span
    _, new_tokens = aligner_fn(" ".join(text[sorted_indices[0][1] : sorted_indices[1][0]]))
    current_tokenization.extend(new_tokens)
    new_span2start = len(current_tokenization)
    _, span_tokens = aligner_fn(" ".join(text[sorted_indices[1][0] : sorted_indices[1][1]]))
    current_tokenization.extend(span_tokens)
    new_span2end = len(current_tokenization)
    span_mapping[sorted_indices[1][0]] = [new_span2start, new_span2end]

    # save back into record
    _, all_text = aligner_fn(" ".join(text))
    record["targets"][0]["span1"] = span_mapping[record["targets"][0]["span1_index"]]
    record["targets"][0]["span2"] = span_mapping[record["targets"][0]["span2_index"]]
    record["text"] = " ".join(all_text)
    return record


def _map_fn(record, tokenizer_name):
    new_record = realign_spans(record, tokenizer_name)
    return json.dumps(new_record)


def preprocess_winograd(fname, tokenizer_name, worker_pool):
    new_name = fname + ".retokenized." + tokenizer_name
    log.info("Processing file: %s", fname)
    # decompress into list of dictionaries
    inputs = list(pd.read_json(fname, lines=True).T.to_dict().values())
    log.info("  saving to %s", new_name)
    map_fn = functools.partial(_map_fn, tokenizer_name=tokenizer_name)
    with open(new_name, "w") as fd:
        for line in tqdm(worker_pool.imap(map_fn, inputs, chunksize=500), total=len(inputs)):
            fd.write(line)
            fd.write("\n")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="tokenizer_name", type=str, help="Tokenizer name.")
    parser.add_argument("--data_dir", type=str, help="Path to data directory.")
    args = parser.parse_args(args)
    worker_pool = multiprocessing.Pool(2)
    for fname in ["train.jsonl", "val.jsonl", "test_with_labels.jsonl"]:
        fname = args.data_dir + fname
        preprocess_winograd(fname, args.tokenizer_name, worker_pool=worker_pool)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
