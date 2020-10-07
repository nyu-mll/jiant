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
from jiant.tasks.lib.templates.shared import double_sentence_featurize, labels_to_bimap
from jiant.utils.python.io import read_file_lines


@dataclass
class Example(BaseExample):
    guid: str
    text_a: str
    text_b: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text_a=tokenizer.tokenize(self.text_a),
            text_b=tokenizer.tokenize(self.text_b),
            label_id=PawsXTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text_a: List
    text_b: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return double_sentence_featurize(
            guid=self.guid,
            input_tokens_a=self.text_a,
            input_tokens_b=self.text_b,
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


class PawsXTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = ["0", "1"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def __init__(self, name, path_dict, language):
        super().__init__(name=name, path_dict=path_dict)
        self.language = language

    def get_train_examples(self):
        return self._create_examples(lines=read_file_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_file_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_file_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            # Skip the header (first line)
            if i == 0:
                continue
            segments = line.strip().split("\t")
            idx, text_a, text_b, label = segments
            examples.append(
                Example(guid="%s-%s" % (set_type, idx), text_a=text_a, text_b=text_b, label=label)
            )
        return examples
