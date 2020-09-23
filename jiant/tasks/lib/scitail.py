import numpy as np
import pandas as pd
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
from jiant.tasks.lib.templates.shared import labels_to_bimap, double_sentence_featurize


@dataclass
class Example(BaseExample):
    guid: str
    input_premise: str
    input_hypothesis: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_premise=tokenizer.tokenize(self.input_premise),
            input_hypothesis=tokenizer.tokenize(self.input_hypothesis),
            label_id=SciTailTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_premise: List
    input_hypothesis: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return double_sentence_featurize(
            guid=self.guid,
            input_tokens_a=self.input_premise,
            input_tokens_b=self.input_hypothesis,
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


class SciTailTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = ["entails", "neutral"]

    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(self.test_path, set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        df = pd.read_csv(path, sep="\t", header=None, names=["premise", "hypothesis", "label"],)
        examples = []
        for i, row in enumerate(df.itertuples()):
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    input_premise=row.premise,
                    input_hypothesis=row.hypothesis,
                    label=row.label if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples
