from collections import Counter

from jiant.tasks.utils import ExclusiveSpan
from jiant.tasks.lib.wic import Example, TokenizedExample
from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer


EXAMPLES = [
    Example(
        guid="train-1",
        sentence1="Approach a task.",
        sentence2="To approach the city.",
        word="approach",
        span1=ExclusiveSpan(start=0, end=8),
        span2=ExclusiveSpan(start=3, end=11),
        label=False,
    ),
    Example(
        guid="train-2",
        sentence1="In England they call takeout food 'takeaway'.",
        sentence2="If you're hungry, there's a takeaway just around the corner.",
        word="takeaway",
        span1=ExclusiveSpan(start=35, end=43),
        span2=ExclusiveSpan(start=28, end=36),
        label=True,
    ),
]

TOKENIZED_EXAMPLES = [
    TokenizedExample(
        guid="train-1",
        sentence1_tokens=["Approach", "a", "task."],
        sentence2_tokens=["To", "approach", "the", "city."],
        word=["approach"],
        sentence1_span=ExclusiveSpan(start=0, end=1),
        sentence2_span=ExclusiveSpan(start=1, end=2),
        label_id=0,
    ),
    TokenizedExample(
        guid="train-2",
        sentence1_tokens=["In", "England", "they", "call", "takeout", "food", "'takeaway'."],
        sentence2_tokens=[
            "If",
            "you're",
            "hungry,",
            "there's",
            "a",
            "takeaway",
            "just",
            "around",
            "the",
            "corner.",
        ],
        word=["takeaway"],
        sentence1_span=ExclusiveSpan(start=6, end=7),
        sentence2_span=ExclusiveSpan(start=5, end=6),
        label_id=1,
    ),
]


def test_task_tokenization():
    token_counter = Counter()
    for example in EXAMPLES:
        token_counter.update(example.sentence1.split() + example.sentence2.split())
    token_vocab = list(token_counter.keys())
    tokenizer = SimpleSpaceTokenizer(vocabulary=token_vocab)

    for example, tokenized_example in zip(EXAMPLES, TOKENIZED_EXAMPLES):
        assert example.tokenize(tokenizer).to_dict() == tokenized_example.to_dict()
