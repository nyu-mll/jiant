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
    sentence_question: str
    answer: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            sentence_question=tokenizer.tokenize(self.sentence_question),
            answer=tokenizer.tokenize(self.answer),
            label_id=MCTACOTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    sentence_question: List
    answer: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return double_sentence_featurize(
            guid=self.guid,
            input_tokens_a=self.sentence_question,
            input_tokens_b=self.answer,
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


class MCTACOTask(Task):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = ["yes", "no"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(
            lines=read_file_lines(self.train_path, strip_lines=True), set_type="train"
        )

    def get_val_examples(self):
        return self._create_examples(
            lines=read_file_lines(self.val_path, strip_lines=True), set_type="val"
        )

    def get_test_examples(self):
        return self._create_examples(
            lines=read_file_lines(self.test_path, strip_lines=True), set_type="test"
        )

    @classmethod
    def _create_examples(cls, lines, set_type):
        # noinspection DuplicatedCode
        examples = []
        last_question = ""
        question_count = -1
        for (i, line) in enumerate(lines):
            sentence, question, answer, label, category = line.split("\t")
            if last_question != question:
                question_count += 1
                last_question = question
            examples.append(
                Example(
                    guid="%s-q%s-%s" % (set_type, question_count, i),
                    sentence_question=sentence + question,
                    answer=answer,
                    label=label if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples
