import collections
import json
import logging as log
from typing import Dict, Iterable, Sequence, Union

import numpy as np
import pandas as pd

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def load_lines(filename: str) -> Iterable[str]:
    """ Load text data, yielding each line. """
    with open(filename) as fd:
        for line in fd:
            yield line.strip()


def load_json_data(filename: str) -> Iterable:
    """ Load JSON records, one per line. """
    with open(filename, "r") as fd:
        for line in fd:
            yield json.loads(line)


def write_json_data(filename: str, records: Iterable[Dict]):
    """ Write JSON records, one per line. """
    with open(filename, "w") as fd:
        for record in records:
            fd.write(json.dumps(record))
            fd.write("\n")


def wrap_singleton_string(item: Union[Sequence, str]):
    """ Wrap a single string as a list. """
    if isinstance(item, str):
        # Can't check if iterable, because a string is an iterable of
        # characters, which is not what we want.
        return [item]
    return item


class EdgeProbingDatasetStats(object):
    def __init__(self):
        self._stats = collections.Counter()

    def update(self, record: Dict):
        stats = self._stats

        stats["count"] += 1
        tokens = record["text"].split()
        stats["token.count"] += len(tokens)
        stats["token.count2"] += len(tokens) ** 2  # for computing RMS
        stats["token.max_count"] = max(len(tokens), stats["token.max_count"])

        # Target stats
        targets = record.get("targets", [])
        stats["targets.count"] += len(targets)
        stats["targets.max_count"] = max(len(targets), stats["targets.max_count"])
        for target in targets:
            labels = wrap_singleton_string(target["label"])
            stats["targets.label.count"] += len(labels)
            span1 = target.get("span1", [-1, -1])
            stats["targets.span1.length"] += max(span1) - min(span1)
            span2 = target.get("span2", [-1, -1])
            stats["targets.span2.length"] += max(span2) - min(span2)

    def compute(self, record_iter: Iterable[Dict]):
        for record in record_iter:
            self.update(record)

    def passthrough(self, record_iter: Iterable[Dict]):
        for record in record_iter:
            self.update(record)
            yield record

    def to_series(self, **kw):
        stats = self._stats
        s = pd.Series(kw, dtype=object)
        s["count"] = stats["count"]
        s["token.count"] = stats["token.count"]
        s["token.mean_count"] = stats["token.count"] / stats["count"]
        s["token.rms_count"] = np.sqrt(stats["token.count2"] / stats["count"])
        s["token.max_count"] = stats["token.max_count"]
        s["targets.count"] = stats["targets.count"]
        s["targets.mean_count"] = stats["targets.count"] / stats["count"]
        s["targets.max_count"] = stats["targets.max_count"]
        s["targets.label.count"] = stats["targets.label.count"]
        s["targets.label.mean_count"] = stats["targets.label.count"] / stats["targets.count"]
        s["targets.span1.mean_length"] = stats["targets.span1.length"] / stats["targets.count"]
        s["targets.span2.mean_length"] = stats["targets.span2.length"] / stats["targets.count"]
        return s

    def format(self, **kw):
        s = self.to_series(**kw)
        return "Stats:\n%s\n" % str(s)

    def __str__(self):
        return self.format()


def write_file_and_print_stats(records: Iterable[Dict], target_fname: str):
    """ Write edge probing records to a JSON file, and print dataset stats. """
    stats = EdgeProbingDatasetStats()
    records = stats.passthrough(records)
    write_json_data(target_fname, records)
    log.info("Wrote examples to %s", target_fname)
    log.info(stats.format())
    return stats
