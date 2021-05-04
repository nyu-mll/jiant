import pytest

from transformers import RobertaTokenizer, BertTokenizer

from jiant.tasks.lib.templates.hacky_tokenization_matching import flat_strip

from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer
import jiant.shared.model_resolution as model_resolution


TEST_STRINGS = ["Hi, my name is Bob Roberts."]
FLAT_STRIP_EXPECTED_STRINGS = ["hi,mynameisbobroberts."]


@pytest.mark.parametrize("model_type", ["albert-base-v2", "roberta-base", "bert-base-uncased"])
def test_delegate_flat_strip(model_type):
    tokenizer = model_resolution.resolve_tokenizer_class(model_type.split("-")[0]).from_pretrained(
        model_type
    )
    for test_string, target_string in zip(TEST_STRINGS, FLAT_STRIP_EXPECTED_STRINGS):
        flat_strip_result = flat_strip(
            tokenizer.tokenize(test_string), tokenizer, return_indices=False
        )
        assert flat_strip_result == target_string
