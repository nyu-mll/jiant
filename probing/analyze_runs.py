#!/usr/bin/env python

# Convenience script to score a set of runs from the predictions.
#
# Usage:
#  python analyze_runs.py -i /path/to/experiments/*/run \
#      -o /tmp/stats.tsv [--parallel n]
#
# Output will be a long-form TSV file containing aggregated and per-class
# predictions for each run.

import argparse
import collections
import json
import logging as log
import os
import re
import sys
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

import analysis
from data import utils

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def find_tasks_and_splits(run_path: str) -> List[Tuple[str, str]]:
    """Find tasks and splits for a particular run."""
    matcher = r"([\w-]+)_(train|val|test)\.json"
    matches = []
    for fname in os.listdir(run_path):
        m = re.match(matcher, fname)
        if m is None:
            continue
        matches.append(m.groups())
    if not matches:
        log.warning("Warning: no predictions found for run '%s'", run_path)
    return matches


def get_run_info(run_path: str, log_name="log.log") -> pd.DataFrame:
    """Extract some run information from the log text."""
    log_path = os.path.join(run_path, log_name)
    if not os.path.isfile(log_path):
        log.warning("Warning: no log found for run '%s'", run_path)
        return None
    train_stats = []
    with open(log_path) as fd:
        for line in fd:
            m = re.match(r"Trained ([\w-]+) for (\d+) batches or (.+) epochs\w*", line)
            if m is None:
                continue
            r = dict(zip(["task", "num_steps", "num_epochs"], m.groups()))
            r["run"] = run_path
            r["label"] = "__run_info__"
            train_stats.append(r)
    return pd.DataFrame.from_records(train_stats)


def analyze_run(run_path: str, task: str, split: str) -> pd.DataFrame:
    log.info("Analyzing: '%s' / %s' / '%s'", run_path, task, split)
    preds = analysis.Predictions.from_run(run_path, task, split)
    scores = preds.score_by_label()
    # Add identifiers
    scores.insert(0, "run", value=run_path)
    scores.insert(1, "task", value=task)
    scores.insert(2, "split", value=split)
    return scores


def _analyze_run(item):
    return analyze_run(*item)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="output", type=str, default="", help="Output file (TSV).")
    parser.add_argument("-i", dest="inputs", type=str, nargs="+", help="Input files.")
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of runs to process in parallel."
    )
    args = parser.parse_args(args)

    work_items = []
    run_info = []
    for run_path in args.inputs:
        for task, split in find_tasks_and_splits(run_path):
            work_items.append((run_path, task, split))
        # Global run info from log file.
        run_info.append(get_run_info(run_path))

    all_scores = []
    if args.parallel > 1:
        from multiprocessing import Pool

        log.info("Processing runs in parallel with %d workers", args.parallel)
        log.getLogger().setLevel(log.WARNING)  # hide INFO spam
        pool = Pool(args.parallel)
        for score in tqdm(pool.imap_unordered(_analyze_run, work_items), total=len(work_items)):
            all_scores.append(score)
        log.getLogger().setLevel(log.INFO)  # re-enable
    else:
        for score in tqdm(map(_analyze_run, work_items), total=len(work_items)):
            all_scores.append(score)

    long_scores = pd.concat(run_info + all_scores, axis=0, ignore_index=True, sort=False)

    if args.output:
        log.info("Writing long-form stats table to %s", args.output)
        long_scores.to_csv(args.output, sep="\t")
    else:
        log.info("Stats:\n%s", str(long_scores))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
