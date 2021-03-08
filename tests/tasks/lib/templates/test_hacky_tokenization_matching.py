import pytest

from transformers import RobertaTokenizer, BertTokenizer

from jiant.tasks.lib.templates.hacky_tokenization_matching import delegate_flat_strip

from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer
import jiant.shared.model_resolution as model_resolution


TEST_STRING = "Hi, my name is Bob Roberts."
FLAT_STRIP_EXPECTED = "hi,mynameisbobroberts."


@pytest.mark.parametrize("model_type", ["albert-base-v2", "roberta-base", "bert-base-uncased"])
def test_delegate_flat_strip(model_type):
    tokenizer = model_resolution.resolve_tokenizer_class(model_type.split("-")[0]).from_pretrained(
        model_type
    )
    flat_strip_result = delegate_flat_strip(
        tokenizer.tokenize(TEST_STRING), tokenizer, return_indices=False
    )
    assert flat_strip_result == FLAT_STRIP_EXPECTED
