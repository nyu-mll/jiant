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
from jiant.tasks.lib.templates.shared import (
    construct_single_input_tokens_and_segment_ids,
    create_input_set_from_tokens_and_segments,
    labels_to_bimap,
)
from jiant.utils.python.io import read_file, read_file_lines


@dataclass
class Example(BaseExample):
    guid: str
    is_english: bool
    text: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid, is_english=self.is_english, text_tokens=tokenizer.tokenize(self.text),
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    is_english: bool
    text_tokens: List

    def featurize(self, tokenizer, feat_spec):
        unpadded_inputs = construct_single_input_tokens_and_segment_ids(
            input_tokens=self.text_tokens, tokenizer=tokenizer, feat_spec=feat_spec,
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
            is_english=self.is_english,
            tokens=unpadded_inputs.unpadded_tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    is_english: bool
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    is_english: torch.BoolTensor
    tokens: list


class TatoebaTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.EMBEDDING

    def __init__(self, name, path_dict, language):
        super().__init__(name=name, path_dict=path_dict)
        self.language = language
        self.lang_bimap = labels_to_bimap(["en", language])

    def get_train_examples(self):
        raise RuntimeError("This task does not support train examples")

    def get_val_examples(self):
        eng_examples = self._create_examples(
            lines=read_file_lines(self.path_dict["eng"]), is_english=True, set_type="val",
        )
        other_examples = self._create_examples(
            lines=read_file_lines(self.path_dict["other"]), is_english=False, set_type="val",
        )
        return eng_examples + other_examples

    def get_test_examples(self):
        raise RuntimeError("This task does not support test examples")

    def get_val_labels(self):
        return np.array(
            [int(x) for x in read_file(self.path_dict["labels_path"]).strip().splitlines()]
        )

    @classmethod
    def _create_examples(cls, lines, is_english, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            examples.append(
                Example(guid="%s-%s" % (set_type, i), is_english=is_english, text=line,)
            )
        return examples


def similarity_search(x, y, dim, normalize=False):
    import faiss

    x = x.copy()
    y = y.copy()
    idx = faiss.IndexFlatL2(dim)
    if normalize:
        faiss.normalize_L2(x)
        faiss.normalize_L2(y)
    idx.add(x)
    scores, prediction = idx.search(y, 1)
    return prediction
