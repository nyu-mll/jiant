import numpy as np

import jiant.tasks.lib.mnli as mnli
from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer


def test_mnli_preproc():
    tokenizer = SimpleSpaceTokenizer(vocabulary=[
        "this", "is", "my", "a", "premise", "hypothesis",
    ])
    feat_spec = tokenizer.get_feat_spec(max_seq_length=16)
    mnli_example = mnli.Example(
        guid="test-1",
        input_premise="this is my premise",
        input_hypothesis="this is a hypothesis",
        label="entailment",
    )
    tokenized_example = mnli_example.tokenize(tokenizer=tokenizer)
    assert tokenized_example.input_premise == ["this", "is", "my", "premise"]
    assert tokenized_example.input_hypothesis == ["this", "is", "a", "hypothesis"]
    featurized_example = tokenized_example.featurize(tokenizer=tokenizer, feat_spec=feat_spec)
    assert np.array_equal(
        featurized_example.input_ids,
        [1, 4, 5, 6, 8, 2, 4, 5, 7, 9, 2, 0, 0, 0, 0, 0],
    )
    assert np.array_equal(
        featurized_example.input_mask,
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    )
    assert np.array_equal(
        featurized_example.segment_ids,
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    )
    assert featurized_example.label_id == 1
