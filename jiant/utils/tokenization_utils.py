import re

from typing import Sequence


def bow_tag_tokens(tokens: Sequence[str], bow_tag: str = "<w>"):
    """Applies a beginning of word (BoW) marker to every token in the tokens sequence."""
    return [bow_tag + t for t in tokens]


def eow_tag_tokens(tokens: Sequence[str], eow_tag: str = "</w>"):
    """Applies a end of word (EoW) marker to every token in the tokens sequence."""
    return [t + eow_tag for t in tokens]


def process_wordpiece_tokens(tokens: Sequence[str]):
    return [process_wordpiece_token_for_alignment(token) for token in tokens]


def process_sentencepiece_tokens(tokens: Sequence[str]):
    return [process_sentencepiece_token_for_alignment(token) for token in tokens]


def process_bytebpe_tokens(tokens: Sequence[str]):
    return [process_bytebpe_token_for_alignment(token) for token in tokens]


def process_wordpiece_token_for_alignment(t):
    """Add word boundary markers, removes token prefix (no-space meta-symbol — '##' for BERT)."""
    if t.startswith("##"):
        return re.sub(r"^##", "", t)
    else:
        return "<w>" + t


def process_sentencepiece_token_for_alignment(t):
    """Add word boundary markers, removes token prefix (space meta-symbol)."""
    if t.startswith("▁"):
        return "<w>" + re.sub(r"^▁", "", t)
    else:
        return t


def process_bytebpe_token_for_alignment(t):
    """Add word boundary markers, removes token prefix (space meta-symbol)."""
    if t.startswith("Ġ"):
        return "<w>" + re.sub(r"^Ġ", "", t)
    else:
        return t
