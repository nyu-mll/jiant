"""Tokenization normalization (to improve likelihood of good cross tokenization alignment).

This module provides helpers for normalizing space tokenization and target tokenizations to make
proper alignment of the tokenizations more likely.

Notes:
    * Code is ported from https://github.com/nyu-mll/jiant/blob/master/jiant/utils/retokenize.py

"""

import transformers
from typing import Sequence

from jiant.utils.testing import utils as test_utils
from jiant.shared.model_resolution import resolve_model_arch_tokenizer
from jiant.proj.main.modeling.primary import JiantTransformersModelFactory


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

    if test_utils.is_pytest():
        from jiant.utils.testing.tokenizer import SimpleSpaceTokenizer

        if isinstance(tokenizer, SimpleSpaceTokenizer):
            return space_tokenization, target_tokenization

    model_arch = resolve_model_arch_tokenizer(tokenizer)
    print(model_arch)
    jiant_transformer_model_class = JiantTransformersModelFactory.get_registry()[model_arch]
    (
        modifed_space_tokenization,
        modifed_target_tokenization,
    ) = jiant_transformer_model_class.normalize_tokenizations(
        tokenizer, space_tokenization, target_tokenization
    )

    # safety check: if normalization changed sequence length, alignment is likely to break.
    assert len(modifed_space_tokenization) == len(space_tokenization)
    assert len(modifed_target_tokenization) == len(target_tokenization)

    return modifed_space_tokenization, modifed_target_tokenization
