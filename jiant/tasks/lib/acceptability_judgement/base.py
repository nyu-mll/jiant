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
from jiant.tasks.lib.templates.shared import single_sentence_featurize, labels_to_bimap
from jiant.utils.python.io import read_json


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            label_id=BaseAcceptabilityTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class BaseAcceptabilityTask(Task):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = ["acceptable", "unacceptable"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)
    DATA_PHASE_MAP = {"train": "train", "dev": "val", "test": "test"}

    def __init__(self, name, path_dict, fold: str):
        # Fold should be a string like "fold1"
        super().__init__(name=name, path_dict=path_dict)
        self.fold = fold

    def get_train_examples(self):
        return self._create_examples(set_type="train")

    def get_val_examples(self):
        return self._create_examples(set_type="val")

    def get_test_examples(self):
        return self._create_examples(set_type="test")

    def _create_examples(self, set_type):
        data = read_json(self.path_dict["data"])
        metadata = read_json(self.path_dict["metadata"])
        assert len(data) == len(metadata)
        examples = []
        for data_row, metadata_row in zip(data, metadata):
            row_phase = self.DATA_PHASE_MAP[metadata_row["misc"][self.fold]]
            if row_phase != set_type:
                continue
            examples.append(
                Example(
                    guid=data_row["pair-id"], text=data_row["context"], label=data_row["label"],
                )
            )
        return examples
