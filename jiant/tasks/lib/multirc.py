import collections
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
from jiant.tasks.lib.templates.shared import (
    labels_to_bimap,
    add_cls_token,
    create_input_set_from_tokens_and_segments,
)
from jiant.tasks.utils import truncate_sequences
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(BaseExample):
    guid: str
    paragraph: str
    question: str
    answer: str
    label: str
    question_id: int

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            paragraph=tokenizer.tokenize(self.paragraph),
            question=tokenizer.tokenize(self.question),
            answer=tokenizer.tokenize(self.answer),
            label_id=MultiRCTask.LABEL_TO_ID[self.label],
            question_id=self.question_id,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    paragraph: List
    question: List
    answer: List
    label_id: int
    question_id: int

    def featurize(self, tokenizer, feat_spec):

        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 4
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 3

        paragraph = truncate_sequences(
            tokens_ls=[self.paragraph],
            max_length=(
                feat_spec.max_seq_length
                - special_tokens_count
                - len(self.question)
                - len(self.answer)
            ),
        )[0]
        unpadded_inputs = add_cls_token(
            unpadded_tokens=(
                paragraph
                + self.question
                + [tokenizer.sep_token]
                + maybe_extra_sep
                + self.answer
                + [tokenizer.sep_token]
            ),
            unpadded_segment_ids=(
                [feat_spec.sequence_a_segment_id] * len(paragraph)
                + [feat_spec.sequence_a_segment_id] * (len(self.question) + 1)
                + maybe_extra_sep_segment_id
                + [feat_spec.sequence_b_segment_id] * (len(self.answer) + 1)
            ),
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
            label_id=self.label_id,
            tokens=unpadded_inputs.unpadded_tokens,
            question_id=self.question_id,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list
    question_id: int


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class MultiRCTask(SuperGlueMixin, Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = [0, 1]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def __init__(self, name, path_dict):
        super().__init__(name=name, path_dict=path_dict)
        self.name = name
        self.path_dict = path_dict

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    def _create_examples(self, lines, set_type):
        examples = []
        for line in lines:
            passage = line["passage"]["text"]
            passage_id = line["idx"]
            for question_dict in line["passage"]["questions"]:
                question = question_dict["question"]
                question_id = question_dict["idx"]
                for answer_dict in question_dict["answers"]:
                    answer_id = answer_dict["idx"]
                    answer = answer_dict["text"]
                    examples.append(
                        Example(
                            # NOTE: MultiRCTask.super_glue_format_preds() is
                            # dependent on this guid format.
                            guid="%s-%s-%s-%s" % (set_type, passage_id, question_id, answer_id),
                            paragraph=passage,
                            question=question,
                            answer=answer,
                            label=answer_dict["label"] if set_type != "test" else self.LABELS[-1],
                            question_id=question_id,
                        )
                    )
        return examples

    @staticmethod
    def super_glue_format_preds(pred_dict):
        """Reformat this task's raw predictions to have the structure expected
        by SuperGLUE.
        """
        lines = []
        # formatting code adapted from: https://github.com/nyu-mll/jiant/blob/
        # 14fae87d2ebc5a45dbe7254e9007d1a148dd6b18/jiant/evaluate.py#L427
        par_qst_ans_d = collections.defaultdict(lambda: collections.defaultdict(list))
        for pred, guid in zip(list(pred_dict["preds"]), list(pred_dict["guids"])):
            passage_id, question_id, answer_id = [int(i) for i in guid.split("-")[1:]]
            ans_d = {"idx": answer_id, "label": int(pred)}
            par_qst_ans_d[passage_id][question_id].append(ans_d)
        for par_idx, qst_ans_d in par_qst_ans_d.items():
            qst_ds = []
            for qst_idx, answers in qst_ans_d.items():
                qst_d = {"idx": qst_idx, "answers": answers}
                qst_ds.append(qst_d)
            out_d = {"idx": par_idx, "passage": {"questions": qst_ds}}
            lines.append(out_d)
        return lines
