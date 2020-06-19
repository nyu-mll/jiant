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
from jiant.tasks.utils import truncate_sequences, ExclusiveSpan
from jiant.utils.python.io import read_json_lines
from jiant.tasks.lib.templates import hacky_tokenization_matching as tokenization_utils


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
        # clean-up
        text = self.text.replace("\n", " ")
        span1_text = self.span1_text.replace("\n", " ")
        span2_text = self.span2_text.replace("\n", " ")

        space_tokens = text.strip().split()
        word1 = space_tokens[self.span1_idx]
        word2 = space_tokens[self.span2_idx]
        assert word1.lower() in text.lower()
        assert word2.lower() in text.lower()
        assert text.strip() == " ".join(space_tokens)
        char_span1 = extract_char_span(text, span1_text, self.span1_idx)
        char_span2 = extract_char_span(text, span2_text, self.span2_idx)
        if self.guid != "val-42":  # Yes, there's an error in this example
            assert text[slice(*char_span1)].lower() == span1_text.lower()
            assert text[slice(*char_span2)].lower() == span2_text.lower()

        tokens1, span1_span = tokenization_utils.get_token_span(
            sentence=text, span=char_span1, tokenizer=tokenizer,
        )
        tokens2, span2_span = tokenization_utils.get_token_span(
            sentence=text, span=char_span2, tokenizer=tokenizer,
        )
        assert tokens1 == tokens2
        return TokenizedExample(
            guid=self.guid,
            tokens=tokens1,
            span1_span=span1_span,
            span2_span=span2_span,
            span1_text=self.span1_text,
            span2_text=self.span2_text,
            label_id=WSCTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    tokens: List
    span1_span: List
    span2_span: List
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


class WSCTask(Task):
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
                    guid="%s-%s" % (set_type, line["idx"]),
                    text=line["text"],
                    span1_idx=line["target"]["span1_index"],
                    span2_idx=line["target"]["span2_index"],
                    span1_text=line["target"]["span1_text"],
                    span2_text=line["target"]["span2_text"],
                    label=line["label"] if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples


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
