import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import (
    labels_to_bimap,
    add_cls_token,
    create_input_set_from_tokens_and_segments,
)
from jiant.tasks.utils import truncate_sequences
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(BaseExample):
    guid: str
    input_obs1: str
    input_hyp1: str
    input_hyp2: str
    input_obs2: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_obs1=tokenizer.tokenize(self.input_obs1),
            input_hyp1=tokenizer.tokenize(self.input_hyp1),
            input_hyp2=tokenizer.tokenize(self.input_hyp2),
            input_obs2=tokenizer.tokenize(self.input_obs2),
            label_id=AbductiveNliTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_obs1: List
    input_hyp1: List
    input_hyp2: List
    input_obs2: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 6  # CLS, SEP-SEP, SEP-SEP, SEP
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 4  # CLS, SEP, SEP, SEP

        input_obs1_a, input_hyp1_a, input_obs2_a = truncate_sequences(
            tokens_ls=[self.input_obs1, self.input_hyp1, self.input_obs2],
            max_length=feat_spec.max_seq_length - special_tokens_count - 1,
            # -1 for self.question
        )
        input_obs1_b, input_hyp2_b, input_obs2_b = truncate_sequences(
            tokens_ls=[self.input_obs1, self.input_hyp2, self.input_obs2],
            max_length=feat_spec.max_seq_length - special_tokens_count - 1,
            # -1 for self.question
        )

        unpadded_inputs_1 = add_cls_token(
            unpadded_tokens=(
                input_obs1_a
                + [tokenizer.sep_token]
                + maybe_extra_sep
                + input_hyp1_a
                + [tokenizer.sep_token]
                + maybe_extra_sep
                + input_obs2_a
                + [tokenizer.sep_token]
            ),
            unpadded_segment_ids=(
                # question + sep(s)
                [feat_spec.sequence_a_segment_id] * (len(input_obs1_a) + 1)
                + maybe_extra_sep_segment_id
                # premise + sep(s)
                + [feat_spec.sequence_a_segment_id] * (len(input_hyp1_a) + 1)
                + maybe_extra_sep_segment_id
                # choice + sep
                + [feat_spec.sequence_b_segment_id] * (len(input_obs2_a) + 1)
            ),
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        unpadded_inputs_2 = add_cls_token(
            unpadded_tokens=(
                input_obs1_b
                + [tokenizer.sep_token]
                + maybe_extra_sep
                + input_hyp2_b
                + [tokenizer.sep_token]
                + maybe_extra_sep
                + input_obs2_b
                + [tokenizer.sep_token]
            ),
            unpadded_segment_ids=(
                # question + sep(s)
                [feat_spec.sequence_a_segment_id] * (len(input_obs1_b) + 1)
                + maybe_extra_sep_segment_id
                # premise + sep(s)
                + [feat_spec.sequence_a_segment_id] * (len(input_hyp2_b) + 1)
                + maybe_extra_sep_segment_id
                # choice + sep
                + [feat_spec.sequence_b_segment_id] * (len(input_obs2_b) + 1)
            ),
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        input_set1 = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs_1.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs_1.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        input_set2 = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs_2.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs_2.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        return DataRow(
            guid=self.guid,
            input_ids=np.stack([input_set1.input_ids, input_set2.input_ids]),
            input_mask=np.stack([input_set1.input_mask, input_set2.input_mask]),
            segment_ids=np.stack([input_set1.segment_ids, input_set2.segment_ids]),
            label_id=self.label_id,
            tokens1=unpadded_inputs_1.unpadded_tokens,
            tokens2=unpadded_inputs_2.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray  # Multiple
    input_mask: np.ndarray  # Multiple
    segment_ids: np.ndarray  # Multiple
    label_id: int
    tokens1: list
    tokens2: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens1: list
    tokens2: list


class AbductiveNliTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.MULTIPLE_CHOICE
    NUM_CHOICES = 2
    LABELS = [1, 2]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(
            lines=read_json_lines(self.path_dict["train_inputs"]),
            labels=self._read_labels(self.path_dict["train_labels"]),
            set_type="train",
        )

    def get_val_examples(self):
        return self._create_examples(
            lines=read_json_lines(self.path_dict["val_inputs"]),
            labels=self._read_labels(self.path_dict["val_labels"]),
            set_type="val",
        )

    def get_test_examples(self):
        raise NotImplementedError()

    @classmethod
    def _create_examples(cls, lines, labels, set_type):
        examples = []
        for (i, (line, label)) in enumerate(zip(lines, labels)):
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    input_obs1=line["obs1"],
                    input_hyp1=line["hyp1"],
                    input_hyp2=line["hyp2"],
                    input_obs2=line["obs2"],
                    label=label,
                )
            )
        return examples

    @classmethod
    def _read_labels(cls, path):
        with open(path) as f:
            return [int(i) for i in f.read().split()]
