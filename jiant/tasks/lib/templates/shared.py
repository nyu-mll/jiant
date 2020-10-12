import numpy as np
from dataclasses import dataclass
from typing import List, NamedTuple

from jiant.tasks.core import FeaturizationSpec
from jiant.tasks.utils import truncate_sequences, pad_to_max_seq_length
from jiant.utils.python.datastructures import BiMap


class Span(NamedTuple):
    start: int
    end: int  # Use exclusive end, for consistency

    def add(self, i: int):
        return Span(start=self.start + i, end=self.end + i)

    def to_slice(self):
        return slice(*self)

    def to_array(self):
        return np.array([self.start, self.end])


@dataclass
class UnpaddedInputs:
    unpadded_tokens: List
    unpadded_segment_ids: List
    cls_offset: int


@dataclass
class InputSet:
    input_ids: List
    input_mask: List
    segment_ids: List


def single_sentence_featurize(
    guid: str,
    input_tokens: List[str],
    label_id: int,
    tokenizer,
    feat_spec: FeaturizationSpec,
    data_row_class,
):
    unpadded_inputs = construct_single_input_tokens_and_segment_ids(
        input_tokens=input_tokens, tokenizer=tokenizer, feat_spec=feat_spec,
    )
    return create_generic_data_row_from_tokens_and_segments(
        guid=guid,
        unpadded_tokens=unpadded_inputs.unpadded_tokens,
        unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
        label_id=label_id,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
        data_row_class=data_row_class,
    )


def double_sentence_featurize(
    guid: str,
    input_tokens_a: List[str],
    input_tokens_b: List[str],
    label_id: int,
    tokenizer,
    feat_spec: FeaturizationSpec,
    data_row_class,
):
    """Featurize an example for a two-input/two-sentence task, and return the example as a DataRow.

    Args:
        guid (str): human-readable identifier for interpretability and debugging.
        input_tokens_a (List[str]): sequence of tokens in segment a.
        input_tokens_b (List[str]): sequence of tokens in segment b.
        label_id (int): int representing the label for the task.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.
        data_row_class (DataRow): DataRow class used in the task.

    Returns:
        DataRow representing an example.

    """
    unpadded_inputs = construct_double_input_tokens_and_segment_ids(
        input_tokens_a=input_tokens_a,
        input_tokens_b=input_tokens_b,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )

    return create_generic_data_row_from_tokens_and_segments(
        guid=guid,
        unpadded_tokens=unpadded_inputs.unpadded_tokens,
        unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
        label_id=label_id,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
        data_row_class=data_row_class,
    )


def construct_single_input_tokens_and_segment_ids(
    input_tokens: List[str], tokenizer, feat_spec: FeaturizationSpec
):
    special_tokens_count = 2  # CLS, SEP

    (input_tokens,) = truncate_sequences(
        tokens_ls=[input_tokens], max_length=feat_spec.max_seq_length - special_tokens_count,
    )

    return add_cls_token(
        unpadded_tokens=input_tokens + [tokenizer.sep_token],
        unpadded_segment_ids=(
            [feat_spec.sequence_a_segment_id]
            + [feat_spec.sequence_a_segment_id] * (len(input_tokens))
        ),
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )


def construct_double_input_tokens_and_segment_ids(
    input_tokens_a: List[str], input_tokens_b: List[str], tokenizer, feat_spec: FeaturizationSpec
):
    """Create token and segment id sequences, apply truncation, add separator and class tokens.

    Args:
        input_tokens_a (List[str]): sequence of tokens in segment a.
        input_tokens_b (List[str]): sequence of tokens in segment b.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        UnpaddedInputs: unpadded inputs with truncation applied and special tokens appended.

    """
    if feat_spec.sep_token_extra:
        maybe_extra_sep = [tokenizer.sep_token]
        maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
        special_tokens_count = 4  # CLS, SEP-SEP, SEP
    else:
        maybe_extra_sep = []
        maybe_extra_sep_segment_id = []
        special_tokens_count = 3  # CLS, SEP, SEP

    input_tokens_a, input_tokens_b = truncate_sequences(
        tokens_ls=[input_tokens_a, input_tokens_b],
        max_length=feat_spec.max_seq_length - special_tokens_count,
    )

    unpadded_tokens = (
        input_tokens_a
        + [tokenizer.sep_token]
        + maybe_extra_sep
        + input_tokens_b
        + [tokenizer.sep_token]
    )
    unpadded_segment_ids = (
        [feat_spec.sequence_a_segment_id] * len(input_tokens_a)
        + [feat_spec.sequence_a_segment_id]
        + maybe_extra_sep_segment_id
        + [feat_spec.sequence_b_segment_id] * len(input_tokens_b)
        + [feat_spec.sequence_b_segment_id]
    )
    return add_cls_token(
        unpadded_tokens=unpadded_tokens,
        unpadded_segment_ids=unpadded_segment_ids,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )


def add_cls_token(
    unpadded_tokens: List[str],
    unpadded_segment_ids: List[int],
    tokenizer,
    feat_spec: FeaturizationSpec,
):
    """Add class token to unpadded inputs.

    Applies class token to end (or start) of unpadded inputs depending on FeaturizationSpec.

    Args:
        unpadded_tokens (List[str]): sequence of unpadded token strings.
        unpadded_segment_ids (List[str]): sequence of unpadded segment ids.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        UnpaddedInputs: unpadded inputs with class token appended.

    """
    if feat_spec.cls_token_at_end:
        return UnpaddedInputs(
            unpadded_tokens=unpadded_tokens + [tokenizer.cls_token],
            unpadded_segment_ids=unpadded_segment_ids + [feat_spec.cls_token_segment_id],
            cls_offset=0,
        )
    else:
        return UnpaddedInputs(
            unpadded_tokens=[tokenizer.cls_token] + unpadded_tokens,
            unpadded_segment_ids=[feat_spec.cls_token_segment_id] + unpadded_segment_ids,
            cls_offset=1,
        )


def create_generic_data_row_from_tokens_and_segments(
    guid: str,
    unpadded_tokens: List[str],
    unpadded_segment_ids: List[int],
    label_id: int,
    tokenizer,
    feat_spec: FeaturizationSpec,
    data_row_class,
):
    """Creates an InputSet and wraps the InputSet into a DataRow class.

    Args:
        guid (str): human-readable identifier (for interpretability and debugging).
        unpadded_tokens (List[str]): sequence of unpadded token strings.
        unpadded_segment_ids (List[int]): sequence of unpadded segment ids.
        label_id (int): int representing the label for the task.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.
        data_row_class (DataRow): data row class to wrap and return the inputs.

    Returns:
        DataRow: data row class containing model inputs.

    """
    input_set = create_input_set_from_tokens_and_segments(
        unpadded_tokens=unpadded_tokens,
        unpadded_segment_ids=unpadded_segment_ids,
        tokenizer=tokenizer,
        feat_spec=feat_spec,
    )
    return data_row_class(
        guid=guid,
        input_ids=np.array(input_set.input_ids),
        input_mask=np.array(input_set.input_mask),
        segment_ids=np.array(input_set.segment_ids),
        label_id=label_id,
        tokens=unpadded_tokens,
    )


def create_input_set_from_tokens_and_segments(
    unpadded_tokens: List[str],
    unpadded_segment_ids: List[int],
    tokenizer,
    feat_spec: FeaturizationSpec,
):
    """Create padded inputs for model.

    Converts tokens to ids, makes input set (input ids, input mask, and segment ids), adds padding.

    Args:
        unpadded_tokens (List[str]): unpadded list of token strings.
        unpadded_segment_ids (List[int]): unpadded list of segment ids.
        tokenizer:
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        Padded input set.

    """
    assert len(unpadded_tokens) == len(unpadded_segment_ids)
    input_ids = tokenizer.convert_tokens_to_ids(unpadded_tokens)
    input_mask = [1] * len(input_ids)
    input_set = pad_features_with_feat_spec(
        input_ids=input_ids,
        input_mask=input_mask,
        unpadded_segment_ids=unpadded_segment_ids,
        feat_spec=feat_spec,
    )
    return input_set


def pad_features_with_feat_spec(
    input_ids: List[int],
    input_mask: List[int],
    unpadded_segment_ids: List[int],
    feat_spec: FeaturizationSpec,
):
    """Apply padding to feature set according to settings from FeaturizationSpec.

    Args:
        input_ids (List[int]): sequence unpadded input ids.
        input_mask (List[int]): unpadded input mask sequence.
        unpadded_segment_ids (List[int]): sequence of unpadded segment ids.
        feat_spec (FeaturizationSpec): Tokenization-related metadata.

    Returns:
        InputSet: input set containing padded input ids, input mask, and segment ids.

    """
    return InputSet(
        input_ids=pad_single_with_feat_spec(
            ls=input_ids, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_id,
        ),
        input_mask=pad_single_with_feat_spec(
            ls=input_mask, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_mask_id,
        ),
        segment_ids=pad_single_with_feat_spec(
            ls=unpadded_segment_ids, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_segment_id,
        ),
    )


def pad_single_with_feat_spec(
    ls: List[int], feat_spec: FeaturizationSpec, pad_idx: int, check=True
):
    """Apply padding to sequence according to settings from FeaturizationSpec.

    Args:
        ls (List[int]): sequence to pad.
        feat_spec (FeaturizationSpec): metadata containing max sequence length and padding settings.
        pad_idx (int): element to use for padding.
        check (bool): True if padded length should be checked as under the max sequence length.

    Returns:
        Sequence with padding applied.

    """
    return pad_to_max_seq_length(
        ls=ls,
        max_seq_length=feat_spec.max_seq_length,
        pad_idx=pad_idx,
        pad_right=not feat_spec.pad_on_left,
        check=check,
    )


def labels_to_bimap(labels):
    """Creates mappings from label to id, and from id to label. See details in docs for BiMap.

    Args:
        labels: sequence of label to map to ids.

    Returns:
        Tuple[Dict, Dict]: mappings from labels to ids, and ids to labels.

    """
    label2id, id2label = BiMap(a=labels, b=list(range(len(labels)))).get_maps()
    return label2id, id2label
