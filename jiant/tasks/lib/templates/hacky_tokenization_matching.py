"""TODO: Remove when Tokenizers gets better  (issue #1189)"""
from jiant.tasks.utils import ExclusiveSpan


def map_tags_to_token_position(flat_stripped, indices, split_text):
    char_index = 0
    current_string = flat_stripped
    positions = [None] * len(split_text)
    for i, token in enumerate(split_text):
        found_index = current_string.find(token.lower())
        assert found_index != -1
        positions[i] = indices[char_index + found_index]
        char_index += found_index + len(token)
        current_string = flat_stripped[char_index:]
    for elem in positions:
        assert elem is not None
    return positions


def convert_mapped_tags(positions, tag_ids, length):
    labels = [None] * length
    mask = [0] * length
    for pos, tag_id in zip(positions, tag_ids):
        labels[pos] = tag_id
        mask[pos] = 1
    return labels, mask


def input_flat_strip(tokens):
    return "".join(tokens).lower()


def flat_strip(tokens, tokenizer, return_indices=False):
    return tokenizer.convert_tokens_to_string(tokens).replace(" ", "").lower()


def starts_with(ls, prefix):
    return ls[: len(prefix)] == prefix


def get_token_span(sentence, span: ExclusiveSpan, tokenizer):
    tokenized = tokenizer.tokenize(sentence)
    tokenized_start1 = tokenizer.tokenize(sentence[: span.start])
    tokenized_start2 = tokenizer.tokenize(sentence[: span.end])
    assert starts_with(tokenized, tokenized_start1)
    # assert starts_with(tokenized, tokenized_start2)  # <- fails because of "does" in "doesn't"
    word = sentence[span.to_slice()]
    assert word.lower().replace(" ", "") in flat_strip(
        tokenized_start2[len(tokenized_start1) :],
        tokenizer=tokenizer,
    )
    token_span = ExclusiveSpan(start=len(tokenized_start1), end=len(tokenized_start2))
    return tokenized, token_span
