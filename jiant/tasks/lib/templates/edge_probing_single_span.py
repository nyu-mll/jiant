"""Single-span edge-probing task base class.

This module defines an AbstractProbingTask and inheritable logic for tokenization and featurization.
Single-span edge-probing tasks should inherit from AbstractProbingTask and the dataclasses
defined here.

"""
from abc import ABC

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import (
    create_input_set_from_tokens_and_segments,
    add_cls_token,
)
from jiant.tasks.utils import ExclusiveSpan, truncate_sequences
from jiant.utils import retokenize
from jiant.utils.tokenization_normalization import normalize_tokenizations


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    span: List[int]
    labels: List[str]

    @property
    def task(self):
        raise NotImplementedError()

    def tokenize(self, tokenizer):
        space_tokenization = self.text.split()
        target_tokenization = tokenizer.tokenize(self.text)
        normed_space_tokenization, normed_target_tokenization = normalize_tokenizations(
            space_tokenization, target_tokenization, tokenizer
        )
        aligner = retokenize.TokenAligner(normed_space_tokenization, normed_target_tokenization)
        target_span = aligner.project_token_span(self.span[0], self.span[1])
        return TokenizedExample(
            guid=self.guid,
            tokens=target_tokenization,
            span=target_span,
            span_text=" ".join(target_tokenization[target_span[0] : target_span[1]]),
            label_ids=[self.task.LABEL_TO_ID[label] for label in self.labels],
            label_num=len(self.task.LABELS),
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    tokens: List[str]
    span: Tuple[int, int]
    span_text: str
    label_ids: List[int]
    label_num: int

    def featurize(self, tokenizer, feat_spec):
        special_tokens_count = 2  # CLS, SEP

        (tokens,) = truncate_sequences(
            tokens_ls=[self.tokens], max_length=feat_spec.max_seq_length - special_tokens_count,
        )

        unpadded_tokens = tokens + [tokenizer.sep_token]
        unpadded_segment_ids = [feat_spec.sequence_a_segment_id] * (len(tokens) + 1)

        unpadded_inputs = add_cls_token(
            unpadded_tokens=unpadded_tokens,
            unpadded_segment_ids=unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        # exclusive spans are converted to inclusive spans for use with SelfAttentiveSpanExtractor
        span = ExclusiveSpan(
            start=self.span[0] + unpadded_inputs.cls_offset,
            end=self.span[1] + unpadded_inputs.cls_offset,
        ).to_inclusive()

        binary_label_ids = np.zeros((self.label_num,), dtype=int)
        for label_id in self.label_ids:
            binary_label_ids[label_id] = 1

        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            spans=np.array([span]),
            label_ids=binary_label_ids,
            tokens=unpadded_inputs.unpadded_tokens,
            span_text=self.span_text,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    spans: np.ndarray
    label_ids: np.ndarray
    tokens: List
    span_text: str


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    spans: torch.LongTensor
    label_ids: torch.LongTensor
    tokens: List
    span_text: List


class AbstractProbingTask(Task, ABC):
    TASK_TYPE = TaskTypes.MULTI_LABEL_SPAN_CLASSIFICATION

    LABELS = NotImplemented
    LABEL_TO_ID = NotImplemented
    ID_TO_LABEL = NotImplemented
