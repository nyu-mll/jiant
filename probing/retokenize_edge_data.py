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
import os
import re
import sys
import multiprocessing
from tqdm import tqdm

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

from src.utils import utils
from src.utils import retokenize

from src.openai_transformer_lm import utils as openai_utils
from pytorch_pretrained_bert import BertTokenizer

from typing import Tuple, List, Text

# For now, this module expects MosesTokenizer as the default.
# TODO: change this once we have better support in core utils.
assert utils.TOKENIZER.__class__.__name__ == "MosesTokenizer"
MosesTokenizer = utils.TOKENIZER

def space_tokenize_with_eow(sentence):
    """Add </w> markers to ensure word-boundary alignment."""
    return [t + "</w>" for t in sentence.split()]

def process_bert_wordpiece_for_alignment(t):
    """Add <w> markers to ensure word-boundary alignment."""
    if t.startswith("##"):
        return re.sub(r"^##", "", t)
    else:
        return "<w>" + t

def space_tokenize_with_bow(sentence):
    """Add <w> markers to ensure word-boundary alignment."""
    return ["<w>" + t for t in sentence.split()]

@functools.lru_cache(maxsize=8, typed=False)
def _get_bert_tokenizer(model_name, do_lower_case):
    log.info(f"Loading BertTokenizer({model_name}, do_lower_case={do_lower_case})")
    return BertTokenizer.from_pretrained(model_name,
                                         do_lower_case=do_lower_case)

##
# Aligner functions. These take a raw string and return a tuple
# of a TokenAligner instance and a list of tokens.
def align_moses(text: Text) -> Tuple[retokenize.TokenAligner, List[Text]]:
    moses_tokens = MosesTokenizer.tokenize(text)
    cleaned_moses_tokens = utils.unescape_moses(moses_tokens)
    ta = retokenize.TokenAligner(text, cleaned_moses_tokens)
    return ta, moses_tokens

def align_openai(text: Text) -> Tuple[retokenize.TokenAligner, List[Text]]:
    eow_tokens = space_tokenize_with_eow(text)
    bpe_tokens = openai_utils.tokenize(text)
    ta = retokenize.TokenAligner(eow_tokens, bpe_tokens)
    return ta, bpe_tokens

def align_bert(text: Text, model_name: str) -> Tuple[retokenize.TokenAligner, List[Text]]:
    # If using lowercase, do this for the source tokens for better matching.
    do_lower_case = model_name.endswith('uncased')
    bow_tokens = space_tokenize_with_bow(text.lower() if do_lower_case else
                                         text)

    bert_tokenizer = _get_bert_tokenizer(model_name, do_lower_case)
    wpm_tokens = bert_tokenizer.tokenize(text)

    # Align using <w> markers for stability w.r.t. word boundaries.
    modified_wpm_tokens = list(map(process_bert_wordpiece_for_alignment,
                                   wpm_tokens))
    ta = retokenize.TokenAligner(bow_tokens, modified_wpm_tokens)
    return ta, wpm_tokens

def get_aligner_fn(tokenizer_name: Text):
    if tokenizer_name == "MosesTokenizer":
        return align_moses
    elif tokenizer_name == "OpenAI.BPE":
        return align_openai
    elif tokenizer_name.startswith("bert-"):
        return functools.partial(align_bert, model_name=tokenizer_name)
    else:
        raise ValueError(f"Unsupported tokenizer '{tokenizer_name}'")

def retokenize_record(record, tokenizer_name):
    """Retokenize an edge probing example. Modifies in-place."""
    text = record['text']
    aligner_fn = get_aligner_fn(tokenizer_name)
    ta, new_tokens = aligner_fn(text)
    record['text'] = " ".join(new_tokens)
    for target in record['targets']:
        if 'span1' in target:
            target['span1'] = list(map(int,
                                       ta.project_span(*target['span1'])))
        if 'span2' in target:
            target['span2'] = list(map(int,
                                       ta.project_span(*target['span2'])))
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
    map_fn = functools.partial(_map_fn,
                               tokenizer_name=tokenizer_name)
    with open(new_name, 'w') as fd:
        for line in tqdm(worker_pool.imap(map_fn, inputs, chunksize=500),
                         total=len(inputs)):
            fd.write(line)
            fd.write("\n")


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest='tokenizer_name', type=str, required=True,
                        help="Tokenizer name.")
    parser.add_argument("--num_parallel", type=int, default=4,
                        help="Number of parallel processes to use.")
    parser.add_argument('inputs', type=str, nargs="+",
                        help="Input JSON files.")
    args = parser.parse_args(args)

    worker_pool = multiprocessing.Pool(args.num_parallel)
    for fname in args.inputs:
        retokenize_file(fname, args.tokenizer_name, worker_pool=worker_pool)


if __name__ == '__main__':
    main(sys.argv[1:])
    sys.exit(0)
