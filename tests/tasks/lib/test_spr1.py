import os
from collections import Counter

import numpy as np
import transformers
from unittest.mock import Mock

from jiant.shared import model_resolution
from jiant.tasks import create_task_from_config_path
from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer


TRAIN_EXAMPLES = [
    {
        "guid": "train-0-0",
        "text": "This conjures up images of a nation full of trim , muscular folks , and suggests "
        "couch potatoes are out of season .",
        "span1": [1, 2],
        "span2": [0, 1],
        "labels": [],
    },
    {
        "guid": "train-0-1",
        "text": "This conjures up images of a nation full of trim , muscular folks , and suggests "
        "couch potatoes are out of season .",
        "span1": [15, 16],
        "span2": [0, 1],
        "labels": ["existed_during", "instigation", "manipulated_by_another"],
    },
    {
        "guid": "train-1-0",
        "text": "`` I spent so much money that if I look at it , and I 'm not on it , I feel guilty"
        " . ''",
        "span1": [2, 3],
        "span2": [1, 2],
        "labels": [
            "awareness",
            "change_of_state",
            "existed_after",
            "existed_before",
            "existed_during",
            "exists_as_physical",
            "instigation",
            "makes_physical_contact",
            "predicate_changed_argument",
            "sentient",
            "volition",
        ],
    },
    {
        "guid": "train-1-1",
        "text": "`` I spent so much money that if I look at it , and I 'm not on it , I feel guilty"
        " . ''",
        "span1": [2, 3],
        "span2": [5, 6],
        "labels": [
            "changes_possession",
            "existed_after",
            "existed_before",
            "existed_during",
            "manipulated_by_another",
        ],
    },
]


TOKENIZED_TRAIN_EXAMPLES = [
    {
        "guid": "train-0-0",
        "tokens": [
            "This",
            "conjures",
            "up",
            "images",
            "of",
            "a",
            "nation",
            "full",
            "of",
            "trim",
            ",",
            "muscular",
            "folks",
            ",",
            "and",
            "suggests",
            "couch",
            "potatoes",
            "are",
            "out",
            "of",
            "season",
            ".",
        ],
        "span1_span": (1, 2),
        "span2_span": (0, 1),
        "span1_text": "conjures",
        "span2_text": "This",
        "label_ids": [],
        "label_num": 18,
    },
    {
        "guid": "train-0-1",
        "tokens": [
            "This",
            "conjures",
            "up",
            "images",
            "of",
            "a",
            "nation",
            "full",
            "of",
            "trim",
            ",",
            "muscular",
            "folks",
            ",",
            "and",
            "suggests",
            "couch",
            "potatoes",
            "are",
            "out",
            "of",
            "season",
            ".",
        ],
        "span1_span": (15, 16),
        "span2_span": (0, 1),
        "span1_text": "suggests",
        "span2_text": "This",
        "label_ids": [8, 10, 13],
        "label_num": 18,
    },
    {
        "guid": "train-1-0",
        "tokens": [
            "``",
            "I",
            "spent",
            "so",
            "much",
            "money",
            "that",
            "if",
            "I",
            "look",
            "at",
            "it",
            ",",
            "and",
            "I",
            "'m",
            "not",
            "on",
            "it",
            ",",
            "I",
            "feel",
            "guilty",
            ".",
            "''",
        ],
        "span1_span": (2, 3),
        "span2_span": (1, 2),
        "span1_text": "spent",
        "span2_text": "I",
        "label_ids": [0, 2, 6, 7, 8, 9, 10, 12, 14, 15, 17],
        "label_num": 18,
    },
    {
        "guid": "train-1-1",
        "tokens": [
            "``",
            "I",
            "spent",
            "so",
            "much",
            "money",
            "that",
            "if",
            "I",
            "look",
            "at",
            "it",
            ",",
            "and",
            "I",
            "'m",
            "not",
            "on",
            "it",
            ",",
            "I",
            "feel",
            "guilty",
            ".",
            "''",
        ],
        "span1_span": (2, 3),
        "span2_span": (5, 6),
        "span1_text": "spent",
        "span2_text": "money",
        "label_ids": [3, 6, 7, 8, 13],
        "label_num": 18,
    },
]


FEATURIZED_TRAIN_EXAMPLE_0 = {
    "guid": "train-0-0",
    "input_ids": np.array(
        [
            1,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            8,
            12,
            13,
            14,
            15,
            13,
            16,
            17,
            18,
            19,
            20,
            21,
            8,
            22,
            23,
            2,
            0,
            0,
        ]
    ),
    "input_mask": np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    ),
    "segment_ids": np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ),
    "label_ids": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    "spans": np.array([[2, 2], [1, 1]]),
    "span1_text": "conjures",
    "span2_text": "This",
    "tokens": [
        "<cls>",
        "This",
        "conjures",
        "up",
        "images",
        "of",
        "a",
        "nation",
        "full",
        "of",
        "trim",
        ",",
        "muscular",
        "folks",
        ",",
        "and",
        "suggests",
        "couch",
        "potatoes",
        "are",
        "out",
        "of",
        "season",
        ".",
        "<sep>",
    ],
}


def test_featurization_of_task_data():
    # Test reading the task-specific toy dataset into examples.
    task = create_task_from_config_path(
        os.path.join(os.path.dirname(__file__), "resources/spr1.json"), verbose=True
    )
    # Test getting train, val, and test examples. Only the contents of train are checked.
    train_examples = task.get_train_examples()
    val_examples = task.get_val_examples()
    for train_example_dataclass, raw_example_dict in zip(train_examples, TRAIN_EXAMPLES):
        assert train_example_dataclass.to_dict() == raw_example_dict
    assert val_examples

    # Testing conversion of examples into tokenized examples
    # the dummy tokenizer requires a vocab — using a Counter here to find that vocab from the data:
    token_counter = Counter()
    for example in train_examples:
        token_counter.update(example.text.split())
    token_vocab = list(token_counter.keys())
    space_tokenizer = SimpleSpaceTokenizer(vocabulary=token_vocab)

    # Mocking to pass normalize_tokenizations's isinstance check during example.tokenize(tokenizer)
    tokenizer = Mock(spec_set=transformers.RobertaTokenizer)
    tokenizer.tokenize.side_effect = space_tokenizer.tokenize
    tokenized_examples = [example.tokenize(tokenizer) for example in train_examples]
    for tokenized_example, expected_tokenized_example in zip(
        tokenized_examples, TOKENIZED_TRAIN_EXAMPLES
    ):
        assert tokenized_example.to_dict() == expected_tokenized_example
    # Dropping the mock and continuing the test with the space tokenizer
    tokenizer = space_tokenizer

    # Testing conversion of a tokenized example to a featurized example
    train_example_0_length = len(tokenized_examples[0].tokens) + 4
    feat_spec = model_resolution.build_featurization_spec(
        model_type="bert-", max_seq_length=train_example_0_length
    )
    featurized_examples = [
        tokenized_example.featurize(tokenizer=tokenizer, feat_spec=feat_spec)
        for tokenized_example in tokenized_examples
    ]
    featurized_example_0_dict = featurized_examples[0].to_dict()

    # not bothering to compare the input_ids because they were made by a dummy tokenizer.
    assert "input_ids" in featurized_example_0_dict
    assert featurized_example_0_dict["guid"] == FEATURIZED_TRAIN_EXAMPLE_0["guid"]
    assert (
        featurized_example_0_dict["input_mask"] == FEATURIZED_TRAIN_EXAMPLE_0["input_mask"]
    ).all()
    assert (
        featurized_example_0_dict["segment_ids"] == FEATURIZED_TRAIN_EXAMPLE_0["segment_ids"]
    ).all()
    assert (featurized_example_0_dict["label_ids"] == FEATURIZED_TRAIN_EXAMPLE_0["label_ids"]).all()
    assert featurized_example_0_dict["tokens"] == FEATURIZED_TRAIN_EXAMPLE_0["tokens"]
    assert featurized_example_0_dict["span1_text"] == FEATURIZED_TRAIN_EXAMPLE_0["span1_text"]
    assert featurized_example_0_dict["span2_text"] == FEATURIZED_TRAIN_EXAMPLE_0["span2_text"]
    assert (featurized_example_0_dict["spans"] == FEATURIZED_TRAIN_EXAMPLE_0["spans"]).all()
