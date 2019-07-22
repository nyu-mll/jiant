#!/usr/bin/env python

# Script to extract final scalar mix weights from a model.
# Supports the weights as used by ELMo, and also those used by
# BertEmbedderModule when used in 'mix' mode.
#
# Usage:
#  python get_scalar_mix.py -i /path/to/experiments/*/run \
#      -o /tmp/scalars.tsv
#
# Output will be a long-form TSV file containing aggregated and per-class
# predictions for each run.

import sys
import os
import re
import json
import collections
import argparse
import glob
import copy
from tqdm import tqdm

import logging as log

import pandas as pd
import torch

from typing import List, Tuple, Iterable, Dict

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def get_scalar(tensor):
    assert tensor.size() == torch.Size([1])
    return tensor.numpy()[0]


def get_mix_scalars(checkpoint_path: str) -> Dict[str, Dict[str, float]]:
    # Load and keep on CPU
    data = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # Find prefixes by matching names.
    gamma_regex = r"^(.+\.scalar_mix(_\d+)?\.)gamma$"
    prefixes = [m.group(1) for m in (re.match(gamma_regex, key) for key in data) if m]
    # Extract scalar set for each prefix
    ret = {}
    for prefix in prefixes:
        s = {}
        for key in data:
            if not key.startswith(prefix):
                continue
            s[key[len(prefix) :]] = get_scalar(data[key])
        ret[prefix] = s
    return ret


def get_run_info(run_path: str, checkpoint_glob: str = "model_state_*.best.th") -> pd.DataFrame:
    """Extract some run information from the log text."""
    checkpoint_paths = glob.glob(os.path.join(run_path, checkpoint_glob))
    checkpoint_paths += glob.glob(os.path.join(run_path, "*", checkpoint_glob))
    if not checkpoint_paths:
        log.warning(f"Warning: no checkpoints found for run {run_path}")

    stats = []
    for checkpoint_path in checkpoint_paths:
        scalars = get_mix_scalars(checkpoint_path)
        for key in scalars:
            r = copy.copy(scalars[key])
            r["run"] = run_path
            r["checkpoint"] = checkpoint_path[len(run_path) :]
            r["scalar_set"] = key
            r["label"] = "__scalar_mix__"
            stats.append(r)
    return pd.DataFrame.from_records(stats)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="output", type=str, default="", help="Output file (TSV).")
    parser.add_argument("-i", dest="inputs", type=str, nargs="+", help="Input files.")
    args = parser.parse_args(args)

    run_info = [get_run_info(path) for path in args.inputs]
    run_info = pd.concat(run_info, axis=0, ignore_index=True, sort=False)

    if args.output:
        log.info("Writing long-form stats table to %s", args.output)
        run_info.to_csv(args.output, sep="\t")
    else:
        log.info("Stats:\n%s", str(run_info))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
