from abc import ABC

import numpy as np
import torch
from dataclasses import dataclass
from typing import List

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
from jiant.tasks.utils import truncate_sequences


@dataclass
class Example(BaseExample):

    guid: str
    prompt: str
    choice_list: List[str]
    label: int

    @property
    def task(self):
        raise NotImplementedError()

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            prompt=tokenizer.tokenize(self.prompt),
            choice_list=[tokenizer.tokenize(choice) for choice in self.choice_list],
            label_id=self.task.CHOICE_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    prompt: List
    choice_list: List[List]
    label_id: int

    def featurize(self, tokenizer, feat_spec):

        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 4  # CLS, SEP-SEP, SEP
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 3  # CLS, SEP, SEP

        input_set_ls = []
        unpadded_inputs_ls = []
        for choice in self.choice_list:
            prompt, choice = truncate_sequences(
                tokens_ls=[self.prompt, choice],
                max_length=feat_spec.max_seq_length - special_tokens_count,
            )
            unpadded_inputs = add_cls_token(
                unpadded_tokens=(
                    # prompt
                    prompt
                    + [tokenizer.sep_token]
                    + maybe_extra_sep
                    # choice
                    + choice
                    + [tokenizer.sep_token]
                ),
                unpadded_segment_ids=(
                    # prompt
                    [feat_spec.sequence_a_segment_id] * (len(prompt) + 1)
                    + maybe_extra_sep_segment_id
                    # choice + sep
                    + [feat_spec.sequence_b_segment_id] * (len(choice) + 1)
                ),
                tokenizer=tokenizer,
                feat_spec=feat_spec,
            )
            input_set = create_input_set_from_tokens_and_segments(
                unpadded_tokens=unpadded_inputs.unpadded_tokens,
                unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
                tokenizer=tokenizer,
                feat_spec=feat_spec,
            )
            input_set_ls.append(input_set)
            unpadded_inputs_ls.append(unpadded_inputs)

        return DataRow(
            guid=self.guid,
            input_ids=np.stack([input_set.input_ids for input_set in input_set_ls]),
            input_mask=np.stack([input_set.input_mask for input_set in input_set_ls]),
            segment_ids=np.stack([input_set.segment_ids for input_set in input_set_ls]),
            label_id=self.label_id,
            tokens_list=[unpadded_inputs.unpadded_tokens for unpadded_inputs in unpadded_inputs_ls],
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray  # Multiple
    input_mask: np.ndarray  # Multiple
    segment_ids: np.ndarray  # Multiple
    label_id: int
    tokens_list: List[List]  # Multiple


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens_list: List


class AbstractMultipleChoiceTask(Task, ABC):

    TASK_TYPE = TaskTypes.MULTIPLE_CHOICE

    CHOICE_KEYS = NotImplemented
    CHOICE_BIMAP = NotImplemented
    NUM_CHOICES = NotImplemented
