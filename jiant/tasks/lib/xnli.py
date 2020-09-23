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
from jiant.utils.python.io import read_jsonl


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
            label_id=XnliTask.LABEL_TO_ID[self.label],
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


class XnliTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = ["contradiction", "entailment", "neutral"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def __init__(self, name, path_dict, language):
        super().__init__(name=name, path_dict=path_dict)
        self.language = language

    def get_train_examples(self):
        raise NotImplementedError()

    def get_val_examples(self):
        return self._create_examples(
            lines=read_jsonl(self.val_path), set_type="val", language=self.language,
        )

    def get_test_examples(self):
        return self._create_examples(
            lines=read_jsonl(self.test_path), set_type="test", language=self.language,
        )

    @classmethod
    def _create_examples(cls, lines, set_type, language):
        examples = []
        for line in lines:
            if line["language"] != language:
                continue
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, len(examples)),
                    input_premise=line["sentence1"],
                    input_hypothesis=line["sentence2"],
                    label=line["gold_label"] if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples
