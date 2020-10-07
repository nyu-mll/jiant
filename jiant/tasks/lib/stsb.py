import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    GlueMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import (
    construct_double_input_tokens_and_segment_ids,
    create_input_set_from_tokens_and_segments,
)
from jiant.utils.python.io import read_jsonl


@dataclass
class Example(BaseExample):
    guid: str
    text_a: str
    text_b: str
    label: float

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text_a=tokenizer.tokenize(self.text_a),
            text_b=tokenizer.tokenize(self.text_b),
            label=self.label,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text_a: List
    text_b: List
    label: float

    def featurize(self, tokenizer, feat_spec):
        # Label not label_id, otherwise we can use double_sentence_featurize
        unpadded_inputs = construct_double_input_tokens_and_segment_ids(
            input_tokens_a=self.text_a,
            input_tokens_b=self.text_b,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
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
            label=self.label,
            tokens=unpadded_inputs.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label: float
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label: torch.FloatTensor
    tokens: list


class StsbTask(GlueMixin, Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.REGRESSION

    def get_train_examples(self):
        return self._create_examples(lines=read_jsonl(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_jsonl(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_jsonl(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    text_a=line["text_a"],
                    text_b=line["text_b"],
                    label=float(line["label"]) if set_type != "test" else 0,
                )
            )
        return examples

    @classmethod
    def get_glue_preds(cls, pred_dict):
        """Returns a tuple of (index, prediction) as expected by GLUE."""
        indexes = []
        predictions = []
        for pred, guid in zip(list(pred_dict["preds"]), list(pred_dict["guids"])):
            indexes.append(int(guid.split("-")[1]))
            predictions.append(str(round(pred, 3)))
        return (indexes, predictions)
