import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    SuperGlueMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import (
    labels_to_bimap,
    add_cls_token,
    create_input_set_from_tokens_and_segments,
)
from jiant.tasks.utils import truncate_sequences, ExclusiveSpan
from jiant.utils import retokenize
from jiant.utils.python.io import read_json_lines
from jiant.utils.tokenization_normalization import normalize_tokenizations


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    span1_idx: int
    span2_idx: int
    span1_text: str
    span2_text: str
    label: str

    def tokenize(self, tokenizer):
        space_tokenization = self.text.split()
        target_tokenization = tokenizer.tokenize(self.text)
        normed_space_tokenization, normed_target_tokenization = normalize_tokenizations(
            space_tokenization, target_tokenization, tokenizer
        )
        aligner = retokenize.TokenAligner(normed_space_tokenization, normed_target_tokenization)
        span1_token_count = len(self.span1_text.split())
        span2_token_count = len(self.span2_text.split())
        target_span1 = ExclusiveSpan(
            *aligner.project_token_span(self.span1_idx, self.span1_idx + span1_token_count)
        )
        target_span2 = ExclusiveSpan(
            *aligner.project_token_span(self.span2_idx, self.span2_idx + span2_token_count)
        )
        return TokenizedExample(
            guid=self.guid,
            tokens=target_tokenization,
            span1_span=target_span1,
            span2_span=target_span2,
            span1_text=self.span1_text,
            span2_text=self.span2_text,
            label_id=WSCTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    tokens: List
    span1_span: ExclusiveSpan
    span2_span: ExclusiveSpan
    span1_text: str
    span2_text: str
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        special_tokens_count = 2  # CLS, SEP

        (tokens,) = truncate_sequences(
            tokens_ls=[self.tokens], max_length=feat_spec.max_seq_length - special_tokens_count,
        )

        unpadded_tokens = tokens + [tokenizer.sep_token]
        unpadded_segment_ids = [feat_spec.sequence_a_segment_id] * (len(self.tokens) + 1)

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
        span1_span = ExclusiveSpan(
            start=self.span1_span[0] + unpadded_inputs.cls_offset,
            end=self.span1_span[1] + unpadded_inputs.cls_offset,
        ).to_inclusive()
        span2_span = ExclusiveSpan(
            start=self.span2_span[0] + unpadded_inputs.cls_offset,
            end=self.span2_span[1] + unpadded_inputs.cls_offset,
        ).to_inclusive()

        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            spans=np.array([span1_span, span2_span]),
            label_id=self.label_id,
            tokens=unpadded_inputs.unpadded_tokens,
            span1_text=self.span1_text,
            span2_text=self.span2_text,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    spans: np.ndarray
    label_id: int
    tokens: List
    span1_text: str
    span2_text: str

    def get_tokens(self):
        return [self.tokens]


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    spans: torch.LongTensor
    label_id: torch.LongTensor
    tokens: List
    span1_text: List
    span2_text: List


class WSCTask(SuperGlueMixin, Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.SPAN_COMPARISON_CLASSIFICATION
    LABELS = [False, True]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    @property
    def num_spans(self):
        return 2

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for line in lines:
            examples.append(
                Example(
                    # NOTE: WSCTask.super_glue_format_preds() is dependent on this guid format.
                    guid="%s-%s" % (set_type, line["idx"]),
                    text=line["text"],
                    span1_idx=line["span1_index"],
                    span2_idx=line["span2_index"],
                    span1_text=line["span1_text"],
                    span2_text=line["span2_text"],
                    label=line["label"] if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples

    @classmethod
    def super_glue_format_preds(cls, pred_dict):
        """Reformat this task's raw predictions to have the structure expected by SuperGLUE."""
        lines = []
        for pred, guid in zip(list(pred_dict["preds"]), list(pred_dict["guids"])):
            lines.append({"idx": int(guid.split("-")[1]), "label": str(cls.LABELS[pred])})
        return lines


def extract_char_span(full_text, span_text, space_index):
    space_tokens = full_text.split()
    extracted_span_text = space_tokens[space_index]
    assert extracted_span_text.lower() in full_text.lower()
    span_length = len(span_text)
    if space_index == 0:
        start = 0
    else:
        start = len(" ".join(space_tokens[:space_index])) + 1
    # exclusive span
    return ExclusiveSpan(start=start, end=start + span_length)
