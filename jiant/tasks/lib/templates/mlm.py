import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple

from jiant.utils.python.datastructures import ReusableGenerator

from jiant.tasks.core import (
    Task,
    TaskTypes,
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
)
from jiant.tasks.lib.templates.shared import (
    construct_single_input_tokens_and_segment_ids,
    create_input_set_from_tokens_and_segments,
)

NON_MASKED_TOKEN_LABEL_ID = -100


@dataclass
class Example(BaseExample):
    guid: str
    text: str

    def tokenize(self, tokenizer):
        return TokenizedExample(guid=self.guid, input_tokens=tokenizer.tokenize(self.text),)


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_tokens: List

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.input_tokens, tokenizer=tokenizer, feat_spec=feat_spec,
        )
        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            # Masking will be performed on the fly in train
            tokens=unpadded_inputs.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    tokens: list


class BaseMLMBatch:
    def get_masked(self, mlm_probability, tokenizer, do_mask):
        raise NotImplementedError


@dataclass
class Batch(BatchMixin, BaseMLMBatch):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    # Only if preset
    tokens: list

    def get_masked(self, mlm_probability, tokenizer, do_mask):
        if do_mask:
            masked_input_ids, masked_lm_labels = mlm_mask_tokens(
                inputs=self.input_ids, tokenizer=tokenizer, mlm_probability=mlm_probability,
            )
            return MaskedBatch(
                masked_input_ids=masked_input_ids,
                input_mask=self.input_mask,
                segment_ids=self.segment_ids,
                masked_lm_labels=masked_lm_labels,
                tokens=self.tokens,
            )
        else:
            # Potentially, the mask_lm_labels should all be -100
            return MaskedBatch(
                masked_input_ids=self.input_ids,
                input_mask=self.input_mask,
                segment_ids=self.segment_ids,
                masked_lm_labels=self.input_ids,
                tokens=self.tokens,
            )


@dataclass
class MaskedBatch(BatchMixin):
    masked_input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    masked_lm_labels: torch.LongTensor
    tokens: list


class MLMTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.MASKED_LANGUAGE_MODELING

    def __init__(self, name, path_dict, mlm_probability=0.15, do_mask=True):
        super().__init__(name=name, path_dict=path_dict)
        self.mlm_probability = mlm_probability
        self.do_mask = do_mask

    def get_train_examples(self):
        return self.create_examples(path=self.train_path, set_type="train", return_generator=True)

    def get_val_examples(self):
        return self.create_examples(path=self.val_path, set_type="val", return_generator=True)

    def get_test_examples(self):
        return self.create_examples(path=self.test_path, set_type="test", return_generator=True)

    @classmethod
    def get_examples_generator(cls, path, set_type):
        with open(path, "r") as f:
            for (i, line) in enumerate(f):
                yield Example(
                    guid="%s-%s" % (set_type, i), text=line.strip(),
                )

    @classmethod
    def create_examples(cls, path, set_type, return_generator):
        generator = ReusableGenerator(cls.get_examples_generator, path=path, set_type=set_type)
        if return_generator:
            return generator
        else:
            return list(generator)


def mlm_mask_tokens(
    inputs: torch.LongTensor, tokenizer, mlm_probability
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """From HuggingFace"""
    device = inputs.device
    inputs = inputs.cpu().clone()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    # noinspection PyProtectedMember
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = NON_MASKED_TOKEN_LABEL_ID  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    # noinspection PyTypeChecker
    return inputs.to(device), labels.to(device)
