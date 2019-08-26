#!/usr/bin/env python

# Helper script to retokenize edge-probing data.
# Uses the given tokenizer, and saves results alongside the original files
# as <fname>.retokenized.<tokenizer_name>
#
# Supported tokenizers:
# - MosesTokenizer
# - OpenAI.BPE: byte-pair-encoding model for OpenAI transformer LM;
#     see https://github.com/openai/finetune-transformer-lm)
# - bert-*: wordpiece models for BERT; see https://arxiv.org/abs/1810.04805
#     and https://github.com/huggingface/pytorch-pretrained-BERT#usage
#
# Usage:
#  python retokenize_edge_data.py <tokenizer_name> /path/to/data/*.json
#
# Speed: takes around 2.5 minutes to process 90000 sentences on a single core.
#
# Note: for OpenAI.BPE, this requires the `spacy` and `ftfy` packages.
# These should only be needed for this script - main.py shouldn't need to do
# any further preprocessing.
#
# Note: for BERT tokenizers, this requires the 'pytorch_pretrained_bert'
# package.

import argparse
import functools
import json
import logging as log
import multiprocessing
import os
import re
import sys
from typing import List, Text, Tuple

from tqdm import tqdm

from pytorch_pretrained_bert import BertTokenizer
from jiant.utils import retokenize, tokenizers, utils

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-t", dest="tokenizer_name", type=str, required=True, help="Tokenizer name.")
PARSER.add_argument(
    "--num_parallel", type=int, default=4, help="Number of parallel processes to use."
)
PARSER.add_argument("inputs", type=str, nargs="+", help="Input JSON files.")

# For now, this module expects MosesTokenizer as the default.
# TODO: change this once we have better support in core utils.
MosesTokenizer = tokenizers.get_tokenizer("MosesTokenizer")
assert MosesTokenizer is not None


def retokenize_record(record, tokenizer_name):
    """Retokenize an edge probing example. Modifies in-place."""
    text = record["text"]
    aligner_fn = retokenize.get_aligner_fn(tokenizer_name)
    ta, new_tokens = aligner_fn(text)
    record["text"] = " ".join(new_tokens)
    for target in record["targets"]:
        if "span1" in target:
            target["span1"] = list(map(int, ta.project_span(*target["span1"])))
        if "span2" in target:
            target["span2"] = list(map(int, ta.project_span(*target["span2"])))
    return record


def _map_fn(line, tokenizer_name):
    record = json.loads(line)
    new_record = retokenize_record(record, tokenizer_name)
    return json.dumps(new_record)


def retokenize_file(fname, tokenizer_name, worker_pool):
    new_name = fname + ".retokenized." + tokenizer_name
    log.info("Processing file: %s", fname)
    inputs = list(utils.load_lines(fname))
    log.info("  saving to %s", new_name)
    map_fn = functools.partial(_map_fn, tokenizer_name=tokenizer_name)
    with open(new_name, "w") as fd:
        for line in tqdm(worker_pool.imap(map_fn, inputs, chunksize=500), total=len(inputs)):
            fd.write(line)
            fd.write("\n")


def main(args):
    args = PARSER.parse_args(args)

    worker_pool = multiprocessing.Pool(args.num_parallel)
    for fname in args.inputs:
        retokenize_file(fname, args.tokenizer_name, worker_pool=worker_pool)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
