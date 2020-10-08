import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from jiant.tasks.lib.templates import mlm as mlm_template
from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
)
from jiant.tasks.utils import ExclusiveSpan, truncate_sequences
from jiant.tasks.lib.templates.shared import (
    construct_single_input_tokens_and_segment_ids,
    create_input_set_from_tokens_and_segments,
    pad_single_with_feat_spec,
)


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    # Spans over char indices
    masked_spans: List[ExclusiveSpan]

    def tokenize(self, tokenizer):
        # masked_tokens will be regular tokens except with tokenizer.mask_token for masked spans
        # label_tokens will be tokenizer.pad_token except with the regular tokens for masked spans
        masked_tokens = []
        label_tokens = []
        curr = 0
        for start, end in self.masked_spans:
            # Handle text before next mask
            tokenized_text = tokenizer.tokenize(self.text[curr:start])
            masked_tokens += tokenized_text
            label_tokens += [tokenizer.pad_token] * len(tokenized_text)

            # Handle mask
            tokenized_masked_text = tokenizer.tokenize(self.text[start:end])
            masked_tokens += [tokenizer.mask_token] * len(tokenized_masked_text)
            label_tokens += tokenized_masked_text
            curr = end
        if curr < len(self.text):
            tokenized_text = tokenizer.tokenize(self.text[curr:])
            masked_tokens += tokenized_text
            label_tokens += [tokenizer.pad_token] * len(tokenized_text)

        return TokenizedExample(
            guid=self.guid, masked_tokens=masked_tokens, label_tokens=label_tokens,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    masked_tokens: List
    label_tokens: List

    def featurize(self, tokenizer, feat_spec):
        # Handle masked_tokens
        unpadded_masked_inputs = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.masked_tokens, tokenizer=tokenizer, feat_spec=feat_spec,
        )
        masked_input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_masked_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_masked_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        # Handle label_tokens
        special_tokens_count = 2  # CLS, SEP
        pad_token = tokenizer.pad_token
        (unpadded_label_tokens,) = truncate_sequences(
            tokens_ls=[self.label_tokens],
            max_length=feat_spec.max_seq_length - special_tokens_count,
        )
        if feat_spec.cls_token_at_end:
            unpadded_label_tokens = unpadded_label_tokens + [pad_token, pad_token]
        else:
            unpadded_label_tokens = [pad_token] + unpadded_label_tokens + [pad_token]
        unpadded_label_token_ids = tokenizer.convert_tokens_to_ids(unpadded_label_tokens)
        masked_lm_labels = pad_single_with_feat_spec(
            ls=unpadded_label_token_ids, feat_spec=feat_spec, pad_idx=feat_spec.pad_token_id,
        )
        masked_lm_labels = np.array(masked_lm_labels)
        masked_lm_labels[
            masked_lm_labels == feat_spec.pad_token_id
        ] = mlm_template.NON_MASKED_TOKEN_LABEL_ID
        return DataRow(
            guid=self.guid,
            masked_input_ids=np.array(masked_input_set.input_ids),
            input_mask=np.array(masked_input_set.input_mask),
            segment_ids=np.array(masked_input_set.segment_ids),
            masked_lm_labels=masked_lm_labels,
            masked_tokens=unpadded_masked_inputs.unpadded_tokens,
            label_tokens=unpadded_label_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    masked_input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    masked_lm_labels: np.ndarray
    masked_tokens: List
    label_tokens: List


@dataclass
class Batch(BatchMixin, mlm_template.BaseMLMBatch):
    masked_input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    masked_lm_labels: torch.LongTensor
    masked_tokens: List
    label_tokens: List

    def get_masked(self, mlm_probability, tokenizer, do_mask):
        assert mlm_probability is None
        assert not do_mask
        return self
