#!/usr/bin/env python

# Helper script to convert edge probing JSON data to TensorFlow Examples in
# TFRecord format.
#
# Usage:
#   python convert_edge_data_to_tfrecord.py /path/to/data/*.json
#
# New files will have the same basename, with .tfrecord extension,
#   e.g. foo_edges.json -> foo_edges.tfrecord

import json
import logging as log
import os
import sys
from typing import Dict, List

import tensorflow as tf
from tqdm import tqdm

from jiant.utils import utils

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def add_string_feature(ex: tf.train.Example, name: str, text: str):
    """Append a single string to the named feature."""
    if isinstance(text, str):
        text = text.encode("utf-8")
    ex.features.feature[name].bytes_list.value.append(text)


def add_ints_feature(ex: tf.train.Example, name: str, ints: List[int]):
    """Append ints from a list to the named feature."""
    ex.features.feature[name].int64_list.value.extend(ints)


def convert_to_example(record: Dict):
    """Convert an edge probing record to a TensorFlow example.

    The example has the following features:
        - text: single string, the text
        - targets.span1: list of int64, alternating start, end indices
        - targets.span2: (optional), list of int64, as targets.span1
        - targets.label: list of strings (see note below)
        - info: single string, serialized info JSON
        - targets.info: list of strings, serialized info JSON for each target

    Due to the limitations of tf.Example, spans are packed into a single flat
    list of length 2*num_targets containing alternating endpoints: [s0, e0, s1,
    e1, ..., sn, en]. You can get individual spans back with tf.reshape(spans,
    [-1, 2]).

    If examples have multiple labels per target (such as for SPR2), these are
    joined into a single string on spaces:
        label: ["foo", "bar", "baz"] -> "foo bar baz"
    You can use tf.string_split and tf.sparse.to_dense to convert these into an
    array of targets.

    Args:
        record: dict, in edge probing record (JSON) format.

    Returns:
        tf.train.Example with features described above.
    """
    ex = tf.train.Example()
    add_string_feature(ex, "text", record["text"])
    add_string_feature(ex, "info", json.dumps(record.get("info", {})))
    for target in record["targets"]:
        label_string = " ".join(utils.wrap_singleton_string(target["label"]))
        add_string_feature(ex, "targets.label", label_string)
        add_ints_feature(ex, "targets.span1", target["span1"])
        if "span2" in target:
            add_ints_feature(ex, "targets.span2", target["span2"])
        add_string_feature(ex, "target.info", json.dumps(target.get("info", {})))

    # Verify that span2 is either empty or aligned to span1.
    num_span1s = len(ex.features.feature["targets.span1"].int64_list.value)
    num_span2s = len(ex.features.feature["targets.span2"].int64_list.value)
    assert num_span2s == num_span1s or num_span2s == 0
    return ex


def convert_file(fname):
    new_name = os.path.splitext(fname)[0] + ".tfrecord"
    log.info("Processing file: %s", fname)
    record_iter = utils.load_json_data(fname)
    log.info("  saving to %s", new_name)
    with tf.python_io.TFRecordWriter(new_name) as writer:
        for record in tqdm(record_iter):
            example = convert_to_example(record)
            writer.write(example.SerializeToString())


def main(args):
    for fname in args:
        convert_file(fname)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
