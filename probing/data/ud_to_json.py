#!/usr/bin/env python

# Script to convert UD English Web Treebank (EWT) data intoa
# edge probing format for dependency parsing.
#
# Usage:
#   ./ud_to_json.py -i <ud_release_dir>/en_ewt-ud-*.conllu \
#       -o /path/to/probing/data/ud_ewt

import argparse
import json
import logging as log
import os
import sys

import conllu
import pandas as pd

import utils

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def convert_ud_file(fd):
    """Convert a UD file to a list of records in edge probing format.

    Args:
        fd: file-like object or list of lines

    Returns:
        list(dict) of edge probing records
    """
    # TODO(Tom): refactor to use conllu to parse file
    prev_line = "FILLER"
    word_lines = []
    examples = []

    for line in fd:
        example_good = 1
        if len(prev_line) < 3:
            spans = []
            words = []

            for word_line in word_lines:
                parts = word_line.split("\t")
                words.append(parts[1].replace('"', '\\"'))
                if "." not in parts[0]:
                    this_id = int(parts[0])
                    this_head = int(parts[6])
                else:
                    example_good = 0
                    this_id = 0
                    this_head = 0
                if this_head == 0:
                    this_head = this_id
                deprel = parts[7]
                spans.append(
                    '{"span1": ['
                    + str(this_id - 1)
                    + ", "
                    + str(this_id)
                    + '], "span2": ['
                    + str(this_head - 1)
                    + ", "
                    + str(this_head)
                    + '], "label": "'
                    + deprel
                    + '"}'
                )

            if example_good:
                examples.append(
                    '{"text": "'
                    + " ".join(words)
                    + '", "targets": ['
                    + ", ".join(spans)
                    + '], "info": {"source": "UD_English-EWT"}}'
                )

            word_lines = []

        elif line[0] != "#" and len(line.strip()) > 1:
            word_lines.append(line)

        prev_line = line.strip()

    # Stopgap: make sure the JSON is valid.
    return [json.loads(e) for e in examples]


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", dest="input_files", type=str, nargs="+", help="Input file(s), e.g. en_ewt-ud-*.conllu"
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory, e.g. /path/to/edges/data/ud_ewt",
    )
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    for filename in args.input_files:
        with open(filename) as fd:
            records = convert_ud_file(fd)
        stats = utils.EdgeProbingDatasetStats()
        records = stats.passthrough(records)
        target_basename = os.path.basename(filename).replace(".conllu", ".json")
        target_fname = os.path.join(args.output_dir, target_basename)
        utils.write_json_data(target_fname, records)
        log.info("Wrote examples to %s", target_fname)
        log.info(stats.format())

    log.info("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
