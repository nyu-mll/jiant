import pytest

import jiant.utils.tokenization_normalization as tn

from transformers import BertTokenizer, XLMTokenizer, RobertaTokenizer, AlbertTokenizer


def test_process_wordpiece_token_sequence():
    expected_adjusted_wordpiece_tokens = [
        "<w>Mr",
        "<w>.",
        "<w>I",
        "mme",
        "lt",
        "<w>chose",
        "<w>to",
        "<w>focus",
        "<w>on",
        "<w>the",
        "<w>in",
        "com",
        "p",
        "re",
        "hen",
        "si",
        "bility",
        "<w>of",
        "<w>accounting",
        "<w>rules",
        "<w>.",
    ]
    original_wordpiece_tokens = [
        "Mr",
        ".",
        "I",
        "##mme",
        "##lt",
        "chose",
        "to",
        "focus",
        "on",
        "the",
        "in",
        "##com",
        "##p",
        "##re",
        "##hen",
        "##si",
        "##bility",
        "of",
        "accounting",
        "rules",
        ".",
    ]
    adjusted_wordpiece_tokens = tn._process_wordpiece_tokens(original_wordpiece_tokens)
    assert adjusted_wordpiece_tokens == expected_adjusted_wordpiece_tokens


def test_process_sentencepiece_token_sequence():
    expected_adjusted_sentencepiece_tokens = [
        "<w>Mr",
        ".",
        "<w>I",
        "m",
        "mel",
        "t",
        "<w>chose",
        "<w>to",
        "<w>focus",
        "<w>on",
        "<w>the",
        "<w>in",
        "comp",
        "re",
        "hen",
        "s",
        "ibility",
        "<w>of",
        "<w>accounting",
        "<w>rules",
        ".",
    ]
    original_sentencepiece_tokens = [
        "▁Mr",
        ".",
        "▁I",
        "m",
        "mel",
        "t",
        "▁chose",
        "▁to",
        "▁focus",
        "▁on",
        "▁the",
        "▁in",
        "comp",
        "re",
        "hen",
        "s",
        "ibility",
        "▁of",
        "▁accounting",
        "▁rules",
        ".",
    ]
    adjusted_sentencepiece_tokens = tn._process_sentencepiece_tokens(original_sentencepiece_tokens)
    assert adjusted_sentencepiece_tokens == expected_adjusted_sentencepiece_tokens


def test_process_bytebpe_token_sequence():
    expected_adjusted_bytebpe_tokens = [
        "Mr",
        ".",
        "<w>Imm",
        "elt",
        "<w>chose",
        "<w>to",
        "<w>focus",
        "<w>on",
        "<w>the",
        "<w>incomp",
        "rehens",
        "ibility",
        "<w>of",
        "<w>accounting",
        "<w>rules",
        ".",
    ]
    original_bytebpe_tokens = [
        "Mr",
        ".",
        "ĠImm",
        "elt",
        "Ġchose",
        "Ġto",
        "Ġfocus",
        "Ġon",
        "Ġthe",
        "Ġincomp",
        "rehens",
        "ibility",
        "Ġof",
        "Ġaccounting",
        "Ġrules",
        ".",
    ]
    adjusted_bytebpe_tokens = tn._process_bytebpe_tokens(original_bytebpe_tokens)
    assert adjusted_bytebpe_tokens == expected_adjusted_bytebpe_tokens


"""
The following tests are marked slow because they load/download real Transformers tokenizers.
These tests will only be run with pytest flag --runslow. TODO: consider mocking tokenizers.
"""


@pytest.mark.slow
def test_space_tokenization_and_bert_uncased_tokenization_normalization():
    text = "Jeff Immelt chose to focus on the incomprehensibility of accounting rules ."
    space_tokenized = text.split(" ")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    target_tokenized = tokenizer.tokenize(text)
    normed_space_tokenized, normed_target_tokenized = tn.normalize_tokenizations(
        space_tokenized, target_tokenized, tokenizer
    )
    assert "".join(normed_space_tokenized) == "".join(normed_target_tokenized)


@pytest.mark.slow
def test_space_tokenization_and_bert_cased_tokenization_normalization():
    text = "Jeff Immelt chose to focus on the incomprehensibility of accounting rules ."
    space_tokenized = text.split(" ")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    target_tokenized = tokenizer.tokenize(text)
    normed_space_tokenized, normed_target_tokenized = tn.normalize_tokenizations(
        space_tokenized, target_tokenized, tokenizer
    )
    assert "".join(normed_space_tokenized) == "".join(normed_target_tokenized)


@pytest.mark.slow
def test_space_tokenization_and_xlm_uncased_tokenization_normalization():
    text = "Jeff Immelt chose to focus on the incomprehensibility of accounting rules ."
    space_tokenized = text.split(" ")
    tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-en-2048")
    target_tokenized = tokenizer.tokenize(text)
    normed_space_tokenized, normed_target_tokenized = tn.normalize_tokenizations(
        space_tokenized, target_tokenized, tokenizer
    )
    assert "".join(normed_space_tokenized) == "".join(normed_target_tokenized)


@pytest.mark.slow
def test_space_tokenization_and_roberta_tokenization_normalization():
    text = "Jeff Immelt chose to focus on the incomprehensibility of accounting rules ."
    space_tokenized = text.split(" ")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    target_tokenized = tokenizer.tokenize(text)
    normed_space_tokenized, normed_target_tokenized = tn.normalize_tokenizations(
        space_tokenized, target_tokenized, tokenizer
    )
    assert "".join(normed_space_tokenized) == "".join(normed_target_tokenized)


@pytest.mark.slow
def test_space_tokenization_and_albert_tokenization_normalization():
    text = "Jeff Immelt chose to focus on the incomprehensibility of accounting rules ."
    space_tokenized = text.split(" ")
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    target_tokenized = tokenizer.tokenize(text)
    normed_space_tokenized, normed_target_tokenized = tn.normalize_tokenizations(
        space_tokenized, target_tokenized, tokenizer
    )
    assert "".join(normed_space_tokenized) == "".join(normed_target_tokenized)


@pytest.mark.slow
def test_normalize_empty_tokenizations():
    text = ""
    space_tokenized = text.split(" ")
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    target_tokenized = tokenizer.tokenize(text)
    with pytest.raises(ValueError):
        tn.normalize_tokenizations(space_tokenized, target_tokenized, tokenizer)


@pytest.mark.slow
def test_space_tokenization_and_unusual_roberta_tokenization_normalization():
    text = (
        "As a practitioner of ethnic humor from the old days on the Borscht Belt , live "
        "television and the nightclub circuit , Mr. Mason instinctively reached for the "
        "vernacular ."
    )
    space_tokenized = text.split(" ")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    target_tokenized = tokenizer.tokenize(text)
    # note 1: exposing the target tokenization to highlight an unusual tokenization:
    # " vernacular" -> 'Ġ', 'vern', 'acular' (the usual pattern suggests 'Ġvern', 'acular')
    assert target_tokenized == [
        "As",
        "Ġa",
        "Ġpractitioner",
        "Ġof",
        "Ġethnic",
        "Ġhumor",
        "Ġfrom",
        "Ġthe",
        "Ġold",
        "Ġdays",
        "Ġon",
        "Ġthe",
        "ĠB",
        "ors",
        "cht",
        "ĠBelt",
        "Ġ,",
        "Ġlive",
        "Ġtelevision",
        "Ġand",
        "Ġthe",
        "Ġnightclub",
        "Ġcircuit",
        "Ġ,",
        "ĠMr",
        ".",
        "ĠMason",
        "Ġinstinctively",
        "Ġreached",
        "Ġfor",
        "Ġthe",
        "Ġ",
        "vern",
        "acular",
        "Ġ.",
    ]
    normed_space_tokenized, normed_target_tokenized = tn.normalize_tokenizations(
        space_tokenized, target_tokenized, tokenizer
    )
    # note: 2: the assert below shows that even with the unusual tokenization (see note 1 above),
    # after normalization the space tokenization and the target tokenization match.
    assert "".join(normed_space_tokenized) == "".join(normed_target_tokenized)
