import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Union

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
    create_input_set_from_tokens_and_segments,
    construct_single_input_tokens_and_segment_ids,
    pad_single_with_feat_spec,
)
from jiant.tasks.lib.templates import hacky_tokenization_matching as tokenization_utils
from jiant.utils.python.io import read_json


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    tag_ids: List[int]

    def tokenize(self, tokenizer):
        tokenized = tokenizer.tokenize(self.text)
        split_text = self.text.split(" ")  # CCG data is space-tokenized
        input_flat_stripped = tokenization_utils.input_flat_strip(split_text)
        flat_stripped, indices = tokenization_utils.delegate_flat_strip(
            tokens=tokenized, tokenizer=tokenizer, return_indices=True,
        )
        assert flat_stripped == input_flat_stripped
        positions = tokenization_utils.map_tags_to_token_position(
            flat_stripped=flat_stripped, indices=indices, split_text=split_text,
        )
        labels, label_mask = tokenization_utils.convert_mapped_tags(
            positions=positions, tag_ids=self.tag_ids, length=len(tokenized),
        )

        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            labels=labels,
            label_mask=label_mask,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    labels: List[Union[int, None]]
    label_mask: List[int]

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.text, tokenizer=tokenizer, feat_spec=feat_spec,
        )
        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )

        # Replicate padding / additional tokens for the label ids and mask
        if feat_spec.sep_token_extra:
            label_suffix = [None, None]
            mask_suffix = [0, 0]
            special_tokens_count = 3  # CLS, SEP-SEP
        else:
            label_suffix = [None]
            mask_suffix = [0]
            special_tokens_count = 2  # CLS, SEP
        unpadded_labels = (
            [None] + self.labels[: feat_spec.max_seq_length - special_tokens_count] + label_suffix
        )
        unpadded_labels = [i if i is not None else -1 for i in unpadded_labels]
        unpadded_label_mask = (
            [0] + self.label_mask[: feat_spec.max_seq_length - special_tokens_count] + mask_suffix
        )

        padded_labels = pad_single_with_feat_spec(
            ls=unpadded_labels, feat_spec=feat_spec, pad_idx=-1,
        )
        padded_label_mask = pad_single_with_feat_spec(
            ls=unpadded_label_mask, feat_spec=feat_spec, pad_idx=0,
        )

        return DataRow(
            guid=self.guid,
            input_ids=np.array(input_set.input_ids),
            input_mask=np.array(input_set.input_mask),
            segment_ids=np.array(input_set.segment_ids),
            label_ids=np.array(padded_labels),
            label_mask=np.array(padded_label_mask),
            tokens=unpadded_inputs.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_ids: np.ndarray
    label_mask: np.ndarray
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_ids: torch.LongTensor
    label_mask: torch.LongTensor
    tokens: list


class CCGTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.TAGGING
    LABELS = range(1363)
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    @property
    def num_labels(self):
        return 1363

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test")

    def get_tags_to_id(self):
        tags_to_id = read_json(self.path_dict["tags_to_id"])
        tags_to_id = {k: int(v) for k, v in tags_to_id.items()}
        return tags_to_id

    def _create_examples(self, path, set_type):
        tags_to_id = self.get_tags_to_id()
        examples = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                text, tags = line.strip().split("\t")
                split_tags = tags.split()
                tag_ids = [tags_to_id[tag] for tag in split_tags]
                examples.append(Example(guid="%s-%s" % (set_type, i), text=text, tag_ids=tag_ids,))
        return examples
