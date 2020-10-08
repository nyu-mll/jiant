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
from jiant.tasks.lib.templates.shared import labels_to_bimap, double_sentence_featurize
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(BaseExample):
    guid: str
    passage_text: str
    query_text: str
    entity_start_char_idx: int  # unused
    entity_end_char_idx: int  # unused
    entity_str: str
    passage_idx: int
    question_idx: int
    answers_dict: dict
    label: bool

    def tokenize(self, tokenizer):
        filled_query_text = self.query_text.replace("@placeholder", self.entity_str)
        return TokenizedExample(
            guid=self.guid,
            passage_tokens=tokenizer.tokenize(self.passage_text),
            query_tokens=tokenizer.tokenize(filled_query_text),
            label_id=ReCoRDTask.LABEL_TO_ID[self.label],
            entity_str=self.entity_str,
            label_set=set(self.answers_dict.values()),
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    passage_tokens: List
    query_tokens: List
    label_id: int
    entity_str: str
    label_set: set

    def featurize(self, tokenizer, feat_spec):
        data_row = double_sentence_featurize(
            guid=self.guid,
            input_tokens_a=self.passage_tokens,
            input_tokens_b=self.query_tokens,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )
        data_row.entity_str = self.entity_str
        data_row.label_set = self.label_set
        return data_row


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list
    entity_str: str
    label_set: set

    def __init__(
        self,
        guid,
        input_ids,
        input_mask,
        segment_ids,
        label_id,
        tokens,
        entity_str=None,
        label_set=None,
    ):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.tokens = tokens
        self.entity_str = entity_str
        self.label_set = label_set


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list
    entity_str: str
    label_set: set


class ReCoRDTask(SuperGlueMixin, Task):
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
            passage_text = line["passage"]["text"]
            for qas in line["qas"]:
                answers_dict = {}
                if set_type != "test":
                    answers_dict = {
                        (answer["start"], answer["end"]): answer["text"]
                        for answer in qas["answers"]
                    }
                for entity in line["passage"]["entities"]:
                    label = False
                    entity_span = (entity["start"], entity["end"])
                    if set_type != "test":
                        if entity_span in answers_dict:
                            assert (
                                passage_text[entity_span[0] : entity_span[1] + 1]
                                == answers_dict[entity_span]
                            )
                            label = True

                    examples.append(
                        Example(
                            # NOTE: ReCoRDTask.super_glue_format_preds() is
                            # dependent on this guid format.
                            guid="%s-%s-%s" % (set_type, len(examples), qas["idx"]),
                            passage_text=passage_text,
                            query_text=qas["query"],
                            entity_start_char_idx=entity_span[0],
                            entity_end_char_idx=entity_span[1] + 1,  # make exclusive
                            entity_str=passage_text[entity_span[0] : entity_span[1] + 1],
                            passage_idx=line["idx"],
                            question_idx=qas["idx"],
                            answers_dict=answers_dict,
                            label=label,
                        )
                    )
        return examples

    @staticmethod
    def super_glue_format_preds(pred_dict):
        return pred_dict["preds"]
