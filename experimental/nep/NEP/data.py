# -*-coding:utf-8-*-
#! /usr/bin/env python

import argparse
import codecs
import json
import os
import sys
import time

import pandas as pd

from NegNN.processors import processor


def to_nli(data):
    """Generates NLI version of the dataset. 
    
    TODO: Add filtering for poor outputs.
    TODO: Figure out reasonable generation techniques.
    """
    def _to_nli():
        for instance in data:
            yield {
                "premise": " ".join(instance["sent"]),
                "hypothesis": " ".join([instance["sent"][i] for i, is_outside_scope in enumerate(instance["labels_idx"]) if is_outside_scope == 0]),
                "label": "contradict",
                "case": "a: scope negated"
            }
            yield {
                "premise": " ".join(instance["sent"]),
                "hypothesis": " ".join([instance["sent"][i] for i, is_outside_scope in enumerate(instance["labels_idx"]) if is_outside_scope == 0 or instance["cues_idx"][i] == 1]),
                "label": "entails",
                "case": "c: both scopes are negated"
            }
            yield {
                "premise": " ".join(instance["sent"]),
                "hypothesis": " ".join([instance["sent"][i] for i, is_outside_scope in enumerate(instance["labels_idx"]) if is_outside_scope != 0 and instance["cues_idx"][i] != 1]),
                "label": "entails",
                "case": "d: non-negated scope is true"
            }
    return list(_to_nli())

def main():

    # Parameters
    # ==================================================

    parser = argparse.ArgumentParser()
    # Model Hyperparameters
    parser.add_argument(
        "--embedding_dim",
        default=50,
        help="Dimensionality of character embedding (default: 50)",
    )
    parser.add_argument(
        "--max_sent_length",
        default=100,
        help="Maximum sentence length for padding (default:100)",
    )
    parser.add_argument(
        "--num_hidden",
        default=200,
        help="Number of hidden units per layer (default:200)",
    )
    parser.add_argument(
        "--num_classes", default=2, help="Number of y classes (default:2)"
    )

    # Training parameters
    parser.add_argument(
        "--num_epochs", default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, help="Learning rate(default: 1e-4)"
    )
    parser.add_argument(
        "--scope_detection",
        default=True,
        help="True if the task is scope detection or joined scope/event detection",
    )
    parser.add_argument(
        "--event_detection",
        default=False,
        help="True is the task is event detection or joined scope/event detection",
    )
    parser.add_argument(
        "--POS_emb",
        default=0,
        help="0: no POS embeddings; 1: normal POS; 2: universal POS",
    )
    parser.add_argument(
        "--emb_update",
        default=False,
        help="True if input embeddings should be updated (default: False)",
    )
    parser.add_argument(
        "--normalize_emb",
        default=False,
        help="True to apply L2 regularization on input embeddings (default: False)",
    )
    # Data Parameters
    parser.add_argument(
        "--test_set",
        default="",
        help="Path to the test filename (to use only in test mode",
    )
    parser.add_argument(
        "--pre_training", default=False, help="True to use pretrained embeddings"
    )
    parser.add_argument(
        "--training_lang",
        default="en",
        help="Language of the tranining data (default: en)",
    )
    FLAGS = parser.parse_args()

    # Data Preparation
    # ==================================================

    fn_training = os.path.abspath("./NegNN/NegNN/data/training/sherlock_train.txt")
    fn_dev = os.path.abspath("./NegNN/NegNN/data/dev/sherlock_dev.txt")

    train_data = processor.load_data(
        fn_training, FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang
    )
    with open("./NEP/raw/train.json", "w") as f:
        json.dump(train_data, f)
    with open("./NEP/nli/train.json", "w") as f:
        json.dump(to_nli(train_data), f)

    dev_data = processor.load_data(
        fn_dev, FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang
    )
    with open("./NEP/raw/dev.json", "w") as f:
        json.dump(dev_data, f)
        pd.DataFrame(dev_data).to_csv("./NEP/raw/dev.tsv", columns=["sent", "tag", "tag_uni", "cue", "scopes_idx", "cues_idx", "label", "labels_idx", "scope",  "tags_idx", "tags_uni_idx"], sep="\t", index=False)
    with open("./NEP/nli/dev.json", "w") as f:
        dev_data = to_nli(dev_data)
        json.dump(dev_data, f, indent=1)
        pd.DataFrame(dev_data).to_csv("./NEP/nli/dev.tsv", columns=["premise", "hypothesis", "label", "case"], sep="\t", index=False)
    tests = [
        "sherlock_cardboard.txt",
        "sherlock_circle.txt",
        "simple_wiki/full/unseen_full.conll",
        "simple_wiki/full/lexical_full.conll",
        "simple_wiki/full/mw_full.conll",
       "simple_wiki/full/prefixal_full.conll",
        "simple_wiki/full/simple_full.conll",
        "simple_wiki/full/suffixal_full.conll",
        "simple_wiki/full/unseen_full.conll",
    ]
    for test_file in tests:
        test_name = test_file.split("/")[-1].split(".")[0]
        fn_test = os.path.abspath(f"./NegNN/NegNN/data/test/{test_file}")
        test_data = processor.load_data(
            fn_test, FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang
        )
        print(f"{test_name} {len(test_data)}")
        with open(f"./NEP/raw/test_{test_name}.json", "w") as f:
            json.dump(test_data, f)
        with open(f"./NEP/nli/test_{test_name}_nli.json", "w") as f:
            json.dump(to_nli(test_data), f)


if __name__ == "__main__":
    main()
