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
from jiant.tasks.lib.templates.shared import double_sentence_featurize, labels_to_bimap
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(BaseExample):
    guid: str
    input_question: str
    input_passage: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_question=tokenizer.tokenize(self.input_question),
            input_passage=tokenizer.tokenize(self.input_passage),
            label_id=BoolQTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_question: List
    input_passage: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return double_sentence_featurize(
            guid=self.guid,
            input_tokens_a=self.input_question,
            input_tokens_b=self.input_passage,
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


class BoolQTask(SuperGlueMixin, Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = [False, True]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

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
                    # NOTE: BoolQTask.super_glue_format_preds() is dependent on this guid format.
                    guid="%s-%s" % (set_type, line["idx"]),
                    input_question=line["question"],
                    input_passage=line["passage"],
                    label=line["label"] if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples

    @classmethod
    def super_glue_format_preds(cls, pred_dict):
        """Reformat this task's raw predictions to have the structure expected by SuperGLUE."""
        lines = []
        for pred, guid in zip(list(pred_dict["preds"]), list(pred_dict["guids"])):
            lines.append({"idx": int(guid.split("-")[1]), "label": str(cls.LABELS[pred]).lower()})
        return lines
