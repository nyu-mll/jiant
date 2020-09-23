"""Tokenization normalization (to improve likelihood of good cross tokenization alignment).

This module provides helpers for normalizing space tokenization and target tokenizations to make
proper alignment of the tokenizations more likely.

Notes:
    * Code is ported from https://github.com/nyu-mll/jiant/blob/master/jiant/utils/retokenize.py

"""

import re
import transformers
from typing import Sequence

from jiant.utils.testing import utils as test_utils


def normalize_tokenizations(
    space_tokenization: Sequence[str],
    target_tokenization: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
):
    """Takes a space tokenization and a known target tokenization and normalizes them for alignment.

    The purpose of this function is to normalize the space-tokenized sequence and target
    tokenization to make proper alignment of the tokenizations more likely. This includes adding
    beginning of word (BoW) or end of word (EoW) tags to the space tokenized and/or target
    tokenization sequences. This also includes removing tokenizer-specific meta symbols, and
    lower casing characters in the space tokenization to match target tokenizers (if uncased).

    Warnings:
        The normalized tokenizations produced by this function are not intended for use as inputs
        to a model. These normalized tokenizations are only intended to be used to help find an
        alignment between space tokenization and target tokenization.

    Args:
        space_tokenization (Seqence[str]): space-tokenized token sequence.
        target_tokenization (Seqence[str]): target tokenizer tokenized sequence.
        tokenizer (PreTrainedTokenizer): tokenizer carrying info needed for target normalization.

    Returns:
        Tuple(Sequence[str], Sequence[str]) of normalized space and target tokenizations.

    Raises:
        ValueError: if either space or target tokenization is an empty sequence.
        ValueError: if tokenizer does not have a normalization strategy in this function.

    """
    if len(space_tokenization) == 0 or len(target_tokenization) == 0:
        raise ValueError("Empty token sequence.")

    if isinstance(tokenizer, transformers.BertTokenizer):
        if tokenizer.init_kwargs.get("do_lower_case", False):
            space_tokenization = [token.lower() for token in space_tokenization]
        modifed_space_tokenization = bow_tag_tokens(space_tokenization)
        modifed_target_tokenization = _process_wordpiece_tokens(target_tokenization)
    elif isinstance(tokenizer, transformers.XLMTokenizer):
        if tokenizer.init_kwargs.get("do_lowercase_and_remove_accent", False):
            space_tokenization = [token.lower() for token in space_tokenization]
        modifed_space_tokenization = eow_tag_tokens(space_tokenization)
        modifed_target_tokenization = target_tokenization
    elif isinstance(tokenizer, transformers.RobertaTokenizer):
        modifed_space_tokenization = bow_tag_tokens(space_tokenization)
        modifed_target_tokenization = ["Ġ" + target_tokenization[0]] + target_tokenization[1:]
        modifed_target_tokenization = _process_bytebpe_tokens(modifed_target_tokenization)
    elif isinstance(tokenizer, (transformers.AlbertTokenizer, transformers.XLMRobertaTokenizer)):
        space_tokenization = [token.lower() for token in space_tokenization]
        modifed_space_tokenization = bow_tag_tokens(space_tokenization)
        modifed_target_tokenization = _process_sentencepiece_tokens(target_tokenization)
    else:
        if test_utils.is_pytest():
            from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer

            if isinstance(tokenizer, SimpleSpaceTokenizer):
                return space_tokenization, target_tokenization
        raise ValueError("Tokenizer not supported.")

    # safety check: if normalization changed sequence length, alignment is likely to break.
    assert len(modifed_space_tokenization) == len(space_tokenization)
    assert len(modifed_target_tokenization) == len(target_tokenization)

    return modifed_space_tokenization, modifed_target_tokenization


def bow_tag_tokens(tokens: Sequence[str], bow_tag: str = "<w>"):
    """Applies a beginning of word (BoW) marker to every token in the tokens sequence."""
    return [bow_tag + t for t in tokens]


def eow_tag_tokens(tokens: Sequence[str], eow_tag: str = "</w>"):
    """Applies a end of word (EoW) marker to every token in the tokens sequence."""
    return [t + eow_tag for t in tokens]


def _process_wordpiece_tokens(tokens: Sequence[str]):
    return [_process_wordpiece_token_for_alignment(token) for token in tokens]


def _process_sentencepiece_tokens(tokens: Sequence[str]):
    return [_process_sentencepiece_token_for_alignment(token) for token in tokens]


def _process_bytebpe_tokens(tokens: Sequence[str]):
    return [_process_bytebpe_token_for_alignment(token) for token in tokens]


def _process_wordpiece_token_for_alignment(t):
    """Add word boundary markers, removes token prefix (no-space meta-symbol — '##' for BERT)."""
    if t.startswith("##"):
        return re.sub(r"^##", "", t)
    else:
        return "<w>" + t


def _process_sentencepiece_token_for_alignment(t):
    """Add word boundary markers, removes token prefix (space meta-symbol)."""
    if t.startswith("▁"):
        return "<w>" + re.sub(r"^▁", "", t)
    else:
        return t


def _process_bytebpe_token_for_alignment(t):
    """Add word boundary markers, removes token prefix (space meta-symbol)."""
    if t.startswith("Ġ"):
        return "<w>" + re.sub(r"^Ġ", "", t)
    else:
        return t
