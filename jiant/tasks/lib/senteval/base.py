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
from jiant.tasks.lib.templates.shared import single_sentence_featurize


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    label: str

    @property
    def label_to_id(self):
        raise NotImplementedError()

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            label_id=self.label_to_id[self.label],
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


class BaseSentEvalTask(Task):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = None  # Override this

    def get_train_examples(self):
        return self._create_examples(set_type="train")

    def get_val_examples(self):
        return self._create_examples(set_type="val")

    def get_test_examples(self):
        return self._create_examples(set_type="test")

    def _create_examples(self, set_type):
        examples = []
        df = pd.read_csv(self.path_dict["data"], sep="\t", names=["phase", "label", "text"])
        phase_key = {"train": "tr", "val": "va", "test": "te"}[set_type]
        sub_df = df[df["phase"] == phase_key]
        for i, row in sub_df.iterrows():
            # noinspection PyArgumentList
            examples.append(
                self.Example(
                    guid="%s-%s" % (set_type, i),
                    text=row.text,
                    label=row.label if set_type != "test" else self.LABELS[-1],
                )
            )
        return examples
