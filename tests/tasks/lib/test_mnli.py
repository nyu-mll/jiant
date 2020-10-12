import os
from collections import Counter
import numpy as np

from jiant.shared import model_resolution
from jiant.tasks import create_task_from_config_path
from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer


TRAIN_EXAMPLES = [
    {
        "guid": "train-0",
        "premise": "Conceptually cream skimming has two basic dimensions - product and geography.",
        "hypothesis": "Product and geography are what make cream skimming work. ",
        "label": "neutral",
    },
    {
        "guid": "train-1",
        "premise": "you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him",
        "hypothesis": "You lose the things to the following level if the people recall.",
        "label": "entailment",
    },
    {
        "guid": "train-2",
        "premise": "One of our number will carry out your instructions minutely.",
        "hypothesis": "A member of my team will execute your orders with immense precision.",
        "label": "entailment",
    },
    {
        "guid": "train-3",
        "premise": "How do you know? All this is their information again.",
        "hypothesis": "This information belongs to them.",
        "label": "entailment",
    },
    {
        "guid": "train-4",
        "premise": "yeah i tell you what though if you go price some of those tennis shoes i can see why now you know they're getting up in the hundred dollar range",
        "hypothesis": "The tennis shoes have a range of prices.",
        "label": "neutral",
    },
]

TOKENIZED_TRAIN_EXAMPLES = [
    {
        "guid": "train-0",
        "premise": [
            "Conceptually",
            "cream",
            "skimming",
            "has",
            "two",
            "basic",
            "dimensions",
            "-",
            "product",
            "and",
            "geography.",
        ],
        "hypothesis": [
            "Product",
            "and",
            "geography",
            "are",
            "what",
            "make",
            "cream",
            "skimming",
            "work.",
        ],
        "label_id": 2,
    },
    {
        "guid": "train-1",
        "premise": [
            "you",
            "know",
            "during",
            "the",
            "season",
            "and",
            "i",
            "guess",
            "at",
            "at",
            "your",
            "level",
            "uh",
            "you",
            "lose",
            "them",
            "to",
            "the",
            "next",
            "level",
            "if",
            "if",
            "they",
            "decide",
            "to",
            "recall",
            "the",
            "the",
            "parent",
            "team",
            "the",
            "Braves",
            "decide",
            "to",
            "call",
            "to",
            "recall",
            "a",
            "guy",
            "from",
            "triple",
            "A",
            "then",
            "a",
            "double",
            "A",
            "guy",
            "goes",
            "up",
            "to",
            "replace",
            "him",
            "and",
            "a",
            "single",
            "A",
            "guy",
            "goes",
            "up",
            "to",
            "replace",
            "him",
        ],
        "hypothesis": [
            "You",
            "lose",
            "the",
            "things",
            "to",
            "the",
            "following",
            "level",
            "if",
            "the",
            "people",
            "recall.",
        ],
        "label_id": 1,
    },
    {
        "guid": "train-2",
        "premise": [
            "One",
            "of",
            "our",
            "number",
            "will",
            "carry",
            "out",
            "your",
            "instructions",
            "minutely.",
        ],
        "hypothesis": [
            "A",
            "member",
            "of",
            "my",
            "team",
            "will",
            "execute",
            "your",
            "orders",
            "with",
            "immense",
            "precision.",
        ],
        "label_id": 1,
    },
    {
        "guid": "train-3",
        "premise": [
            "How",
            "do",
            "you",
            "know?",
            "All",
            "this",
            "is",
            "their",
            "information",
            "again.",
        ],
        "hypothesis": ["This", "information", "belongs", "to", "them."],
        "label_id": 1,
    },
    {
        "guid": "train-4",
        "premise": [
            "yeah",
            "i",
            "tell",
            "you",
            "what",
            "though",
            "if",
            "you",
            "go",
            "price",
            "some",
            "of",
            "those",
            "tennis",
            "shoes",
            "i",
            "can",
            "see",
            "why",
            "now",
            "you",
            "know",
            "they're",
            "getting",
            "up",
            "in",
            "the",
            "hundred",
            "dollar",
            "range",
        ],
        "hypothesis": ["The", "tennis", "shoes", "have", "a", "range", "of", "prices."],
        "label_id": 2,
    },
]

FEATURIZED_TRAIN_EXAMPLE_0 = {
    "guid": "train-0",
    "input_ids": np.array([1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 15, 13, 16, 17, 18, 19, 5, 6, 20, 2]),
    "input_mask": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    "segment_ids": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    "label_id": 2,
    "tokens": [
        "<cls>",
        "Conceptually",
        "cream",
        "skimming",
        "has",
        "two",
        "basic",
        "dimensions",
        "-",
        "<sep>",
        "Product",
        "and",
        "geography",
        "are",
        "what",
        "make",
        "cream",
        "skimming",
        "work.",
        "<sep>",
    ],
}


def test_featurization_of_task_data():
    # Test reading the task-specific toy dataset into examples.
    task = create_task_from_config_path(
        os.path.join(os.path.dirname(__file__), "resources/mnli.json"), verbose=False
    )
    # Test getting train, val, and test examples. Only the contents of train are checked.
    train_examples = task.get_train_examples()
    val_examples = task.get_val_examples()
    test_examples = task.get_test_examples()
    for train_example_dataclass, raw_example_dict in zip(train_examples, TRAIN_EXAMPLES):
        assert train_example_dataclass.to_dict() == raw_example_dict
    assert val_examples
    assert test_examples

    # Testing conversion of examples into tokenized examples
    # the dummy tokenizer requires a vocab — using a Counter here to find that vocab from the data:
    token_counter = Counter()
    for example in train_examples:
        token_counter.update(example.premise.split())
        token_counter.update(example.hypothesis.split())
    token_vocab = list(token_counter.keys())
    tokenizer = SimpleSpaceTokenizer(vocabulary=token_vocab)
    tokenized_examples = [example.tokenize(tokenizer) for example in train_examples]
    for tokenized_example, expected_tokenized_example in zip(
        tokenized_examples, TOKENIZED_TRAIN_EXAMPLES
    ):
        assert tokenized_example.to_dict() == expected_tokenized_example

    # Testing conversion of a tokenized example to a featurized example
    train_example_0_length = len(tokenized_examples[0].premise) + len(
        tokenized_examples[0].hypothesis
    )
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
    assert featurized_example_0_dict["label_id"] == FEATURIZED_TRAIN_EXAMPLE_0["label_id"]
    assert featurized_example_0_dict["tokens"] == FEATURIZED_TRAIN_EXAMPLE_0["tokens"]
