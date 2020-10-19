from abc import ABC

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple

import jiant.shared.model_resolution as model_resolution
from jiant.tasks.core import (
    Task,
    TaskTypes,
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
)
from jiant.tasks.lib.templates.shared import (
    create_input_set_from_tokens_and_segments,
    add_cls_token,
)
from jiant.tasks.utils import truncate_sequences, pad_to_max_seq_length
from jiant.utils.retokenize import TokenAligner


@dataclass
class Example(BaseExample):

    guid: str
    passage: str
    question: str
    answer: str
    answer_char_span: (int, int)

    def tokenize(self, tokenizer):
        passage = (
            self.passage.lower()
            if model_resolution.resolve_is_lower_case(tokenizer=tokenizer)
            else self.passage
        )
        passage_tokens = tokenizer.tokenize(passage)
        token_aligner = TokenAligner(source=passage, target=passage_tokens)
        answer_token_span = token_aligner.project_char_to_token_span(
            self.answer_char_span[0], self.answer_char_span[1], inclusive=True
        )

        return TokenizedExample(
            guid=self.guid,
            passage=passage_tokens,
            question=tokenizer.tokenize(self.question),
            answer_str=self.answer,
            passage_str=passage,
            answer_token_span=answer_token_span,
            token_idx_to_char_idx_map=token_aligner.source_char_idx_to_target_token_idx.T,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    passage: List[str]
    question: List[str]
    answer_str: str
    passage_str: str
    answer_token_span: Tuple[int, int]
    token_idx_to_char_idx_map: np.ndarray

    def featurize(self, tokenizer, feat_spec):

        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 4  # CLS, SEP-SEP, SEP
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 3  # CLS, SEP, SEP

        passage, question = truncate_sequences(
            tokens_ls=[self.passage, self.question],
            max_length=feat_spec.max_seq_length - special_tokens_count,
        )
        assert (
            len(passage) >= self.answer_token_span[1]
        ), f"Answer span {self.answer_token_span} truncated, please raise max_seq_length."
        unpadded_inputs = add_cls_token(
            unpadded_tokens=(
                passage + [tokenizer.sep_token] + maybe_extra_sep + question + [tokenizer.sep_token]
            ),
            unpadded_segment_ids=(
                [feat_spec.sequence_a_segment_id] * (len(passage) + 1)
                + maybe_extra_sep_segment_id
                + [feat_spec.sequence_b_segment_id] * (len(question) + 1)
            ),
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        gt_span_idxs = list(map(lambda x: x + unpadded_inputs.cls_offset, self.answer_token_span))
        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        pred_span_mask = pad_to_max_seq_length(
            ls=[0] * unpadded_inputs.cls_offset + [1] * len(passage),
            max_seq_length=feat_spec.max_seq_length,
            pad_idx=0,
            pad_right=not feat_spec.pad_on_left,
        )
        token_idx_to_char_idx_start = pad_to_max_seq_length(
            ls=[-1] * unpadded_inputs.cls_offset
            + (self.token_idx_to_char_idx_map > 0).argmax(axis=1).tolist()[: len(passage)],
            max_seq_length=feat_spec.max_seq_length,
            pad_idx=-1,
            pad_right=not feat_spec.pad_on_left,
        )
        token_idx_to_char_idx_end = pad_to_max_seq_length(
            ls=[-1] * unpadded_inputs.cls_offset
            + self.token_idx_to_char_idx_map.cumsum(axis=1).argmax(axis=1).tolist()[: len(passage)],
            max_seq_length=feat_spec.max_seq_length,
            pad_idx=-1,
            pad_right=not feat_spec.pad_on_left,
        )
        # When there are multiple greatest elements, argmax will return the index of the first one.
        # So, (x > 0).argmax() will return the index of the first non-zero element in an array,
        # token_idx_to_char_idx_start is computed in this way to map each token index to the
        # beginning char index of that token. On the other side, x.cumsum().argmax() will return
        # the index of the last non-zero element in an array, token_idx_to_char_idx_end is
        # computed in this way to map each token index to ending char index.
        # Once the model predict a span over the token index, these to mapping will help to project
        # the span back to char index, and slice the predicted answer string from the input text.

        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            gt_span_str=self.answer_str,
            gt_span_idxs=np.array(gt_span_idxs),
            selection_str=self.passage_str,
            selection_token_mask=np.array(pred_span_mask),
            token_idx_to_char_idx_start=np.array(token_idx_to_char_idx_start),
            token_idx_to_char_idx_end=np.array(token_idx_to_char_idx_end),
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    gt_span_str: str
    gt_span_idxs: np.ndarray
    selection_str: str
    selection_token_mask: np.ndarray
    token_idx_to_char_idx_start: np.ndarray
    token_idx_to_char_idx_end: np.ndarray


@dataclass
class Batch(BatchMixin):
    guid: List[str]
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    gt_span_str: List[str]
    gt_span_idxs: torch.LongTensor
    selection_str: List[str]
    selection_token_mask: torch.LongTensor
    token_idx_to_char_idx_start: torch.LongTensor
    token_idx_to_char_idx_end: torch.LongTensor


class AbstractSpanPredictionTask(Task, ABC):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch
    TASK_TYPE = TaskTypes.SPAN_PREDICTION
