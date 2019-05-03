import collections
import json
import logging as log
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from tqdm import tqdm

import utils

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def _incl_to_excl(span: Tuple[int, int]):
    return (span[0], span[1] + 1)


def _make_target(label: List[str], span1: Tuple[int, int], span2: Tuple[int, int] = None):
    t = {"span1": _incl_to_excl(span1), "label": label}
    if span2 is not None:
        t["span2"] = _incl_to_excl(span2)
    return t


def make_record(spans, sentence):
    record = {}
    record["info"] = {"document_id": sentence.document_id, "sentence_id": sentence.sentence_id}

    record["text"] = " ".join(sentence.words)
    record["targets"] = [_make_target(*s) for s in spans]
    return record


def constituents_to_record(parse_tree):
    """Function converting Tree object to dictionary compatible with common JSON format
     copied from ptb_process.py so it doesn't have dependencies
    """
    form_function_discrepancies = ["ADV", "NOM"]
    grammatical_rule = ["DTV", "LGS", "PRD", "PUT", "SBJ", "TPC", "VOC"]
    adverbials = ["BNF", "DIR", "EXT", "LOC", "MNR", "PRP", "TMP"]
    miscellaneous = ["CLR", "CLF", "HLN", "TTL"]
    punctuations = ["-LRB-", "-RRB-", "-LCB-", "-RCB-", "-LSB-", "-RSB-"]

    record = {}
    record["text"] = " ".join(parse_tree.flatten())
    record["targets"] = []

    max_height = parse_tree.height()
    for i, leaf in enumerate(parse_tree.subtrees(lambda t: t.height() == 2)):
        # modify the leafs by adding their index in the parse_tree
        leaf[0] = (leaf[0], str(i))

    for index, subtree in enumerate(parse_tree.subtrees()):
        assoc_words = subtree.leaves()
        assoc_words = [(i, int(j)) for i, j in assoc_words]
        assoc_words.sort(key=lambda elem: elem[1])
        tmp_tag_list = subtree.label().replace("=", "-").replace("|", "-").split("-")
        label = tmp_tag_list[0]
        if tmp_tag_list[-1].isdigit():  # Getting rid of numbers at the end of each tag
            fxn_tgs = tmp_tag_list[1:-1]
        else:
            fxn_tgs = tmp_tag_list[1:]
        # Special cases:
        if len(tmp_tag_list) > 1 and tmp_tag_list[1] == "S":  # Case when we have 'PRP-S' or 'WP-S'
            label = tmp_tag_list[0] + "-" + tmp_tag_list[1]
            fxn_tgs = tmp_tag_list[2:-1] if tmp_tag_list[-1].isdigit() else tmp_tag_list[2:]
        if (
            subtree.label() in punctuations
        ):  # Case when we have one of the strange punctions, such as round brackets
            label, fxn_tgs = subtree.label(), []
        target = {"span1": [int(assoc_words[0][1]), int(assoc_words[-1][1]) + 1], "label": label}

        fxn_tgs = set(fxn_tgs)
        target["info"] = {
            "height": subtree.height() - 1,
            #  "depth": find_depth(parse_tree, subtree),
            "form_function_discrepancies": list(fxn_tgs.intersection(form_function_discrepancies)),
            "grammatical_rule": list(fxn_tgs.intersection(grammatical_rule)),
            "adverbials": list(fxn_tgs.intersection(adverbials)),
            "miscellaneous": list(fxn_tgs.intersection(miscellaneous)),
        }
        record["targets"].append(target)

    return record


def find_links(span_list):
    pairs = []
    for i, span1 in enumerate(span_list):
        for span2 in span_list[i + 1 :]:
            pairs.append((str(int(span1[0] == span2[0])), span1[1], span2[1]))
    return pairs


def get_frames(sentence):
    for frame, bio_tags in sentence.srl_frames:
        frame_targets = []
        spans = bio_tags_to_spans(bio_tags)
        head_span = None
        other_spans = []
        for (tag, indices) in spans:
            if tag == "V":
                head_span = indices
            else:
                other_spans.append((tag, indices))
        if head_span is None:
            print(frame, bio_tags)
        for span2_tag, span2 in other_spans:
            frame_targets.append((span2_tag, head_span, span2))
        yield frame_targets


def process_task_split(ontonotes_reader, task: str, stats: collections.Counter):
    for sentence in ontonotes_reader:
        if task == "ner":
            spans = bio_tags_to_spans(sentence.named_entities)
            yield make_record(spans, sentence)
        elif task == "const":
            if sentence.parse_tree is not None:
                record = constituents_to_record(sentence.parse_tree)
                record["info"] = {
                    "document_id": sentence.document_id,
                    "sentence_id": sentence.sentence_id,
                }
                yield record
            else:
                stats["missing_tree"] += 1
                yield make_record([], sentence)
        elif task == "coref":
            spans = find_links(list(sentence.coref_spans))
            yield make_record(spans, sentence)
            stats["num_entities"] += len(sentence.coref_spans)
        elif task == "srl":
            for frame_spans in get_frames(sentence):
                yield make_record(frame_spans, sentence)
                stats["frames"] += 1
        else:
            raise ValueError(f"Unrecognized task '{task}'")

        stats["sentences"] += 1


def main(args):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ontonotes",
        type=str,
        required=True,
        help="Path to OntoNotes, e.g. /path/to/conll-formatted-ontonotes-5.0",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", help="Tasks, one or more of {const, coref, ner, srl}."
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "development", "test", "conll-2012-test"],
        help="Splits, one or more of {train, development, test, conll-2012-test}.",
    )
    parser.add_argument(
        "-o", dest="output_dir", type=str, default=".", help="Output directory for JSON files."
    )
    args = parser.parse_args(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    import pandas as pd

    pd.options.display.float_format = "{:.2f}".format

    # Load OntoNotes reader.
    ontonotes = Ontonotes()
    for split in args.splits:
        for task in args.tasks:
            source_path = os.path.join(args.ontonotes, "data", split)
            ontonotes_reader = ontonotes.dataset_iterator(file_path=source_path)

            log.info("Processing split '%s' for task '%s'", split, task)
            task_dir = os.path.join(args.output_dir, task)
            if not os.path.isdir(task_dir):
                os.mkdir(task_dir)
            target_fname = os.path.join(task_dir, f"{split}.json")
            ontonotes_stats = collections.Counter()
            converted_records = process_task_split(tqdm(ontonotes_reader), task, ontonotes_stats)

            stats = utils.EdgeProbingDatasetStats()
            converted_records = stats.passthrough(converted_records)
            utils.write_json_data(target_fname, converted_records)
            log.info("Wrote examples to %s", target_fname)
            log.info(stats.format())
            log.info(str(pd.Series(ontonotes_stats, dtype=object)))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
