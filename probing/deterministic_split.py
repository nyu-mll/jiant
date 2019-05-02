#!/usr/bin/env python

# Helper script to determinstically, but pseudo-randomly split a file by lines.
# This is easy to to with UNIX tools (shuf and split), but the former may
# behave differently on different systems. For repeatability, we implement this
# using a seeded RNG in Python instead.
#
# Usage, to split 'all.txt' into 'train.txt' and 'dev.txt', with train
# containing approximately 80% of the examples:
#  python deterministic_split.py -s 42 -f 0.8 -i all.txt -o train.txt dev.txt
#
# Note: this /should/ be reproducible in future Python versions, but may
# require a minor code change to use backwards-compatible RNG seeding.
# See https://docs.python.org/3/library/random.html#notes-on-reproducibility
#
# For edge probing experiments, this is run using Python 3.6.8

import argparse
import logging as log
import os
import random
import sys

from tqdm import tqdm

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "-f",
        "--fraction",
        dest="fraction",
        type=float,
        required=True,
        help="Fraction to send to first output file.",
    )
    parser.add_argument("-i", dest="input", type=str, required=True, help="Input file.")
    parser.add_argument(
        "-o", dest="outputs", type=str, required=True, nargs=2, help="Output files."
    )
    args = parser.parse_args(args)
    assert (args.fraction >= 0) and (args.fraction <= 1)

    train_fd = open(args.outputs[0], "w")
    dev_fd = open(args.outputs[1], "w")
    train_ctr = 0
    dev_ctr = 0
    random.seed(args.seed)
    for line in open(args.input, "r"):
        if random.random() <= args.fraction:
            train_fd.write(line)
            train_ctr += 1
        else:
            dev_fd.write(line)
            dev_ctr += 1
    total_ex = train_ctr + dev_ctr
    log.info(f"Read {total_ex} examples from {args.input}")
    train_frac = train_ctr / total_ex
    log.info(f"Train: {train_ctr} examples ({train_frac:.02%})")
    dev_frac = dev_ctr / total_ex
    log.info(f"Dev: {dev_ctr} examples ({dev_frac:.02%})")


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
