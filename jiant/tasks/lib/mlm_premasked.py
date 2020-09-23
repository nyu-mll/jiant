from dataclasses import dataclass
from typing import List

import jiant.utils.python.io as py_io
from jiant.tasks.core import (
    Task,
    TaskTypes,
    BaseExample,
)
from jiant.tasks.utils import ExclusiveSpan
from .templates import mlm_premasked as mlm_premasked_template


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    # Spans over char indices
    masked_spans: List[ExclusiveSpan]

    def tokenize(self, tokenizer):
        # masked_tokens will be regular tokens except with tokenizer.mask_token for masked spans
        # label_tokens will be tokenizer.pad_token except with the regular tokens for masked spans
        masked_tokens = []
        label_tokens = []
        curr = 0
        for start, end in self.masked_spans:
            # Handle text before next mask
            tokenized_text = tokenizer.tokenize(self.text[curr:start])
            masked_tokens += tokenized_text
            label_tokens += [tokenizer.pad_token] * len(tokenized_text)

            # Handle mask
            tokenized_masked_text = tokenizer.tokenize(self.text[start:end])
            masked_tokens += [tokenizer.mask_token] * len(tokenized_masked_text)
            label_tokens += tokenized_masked_text
            curr = end
        if curr < len(self.text):
            tokenized_text = tokenizer.tokenize(self.text[curr:])
            masked_tokens += tokenized_text
            label_tokens += [tokenizer.pad_token] * len(tokenized_text)

        return TokenizedExample(
            guid=self.guid, masked_tokens=masked_tokens, label_tokens=label_tokens,
        )


@dataclass
class TokenizedExample(mlm_premasked_template.TokenizedExample):
    pass


@dataclass
class DataRow(mlm_premasked_template.BaseDataRow):
    pass


@dataclass
class Batch(mlm_premasked_template.Batch):
    pass


class MLMPremaskedTask(Task):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.MASKED_LANGUAGE_MODELING

    def __init__(self, name, path_dict):
        super().__init__(name=name, path_dict=path_dict)
        self.mlm_probability = None
        self.do_mask = False

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        for i, row in enumerate(py_io.read_jsonl(path)):
            yield Example(
                guid="%s-%s" % (set_type, i),
                text=row["text"],
                masked_spans=[ExclusiveSpan(start, end) for start, end in row["masked_spans"]],
            )
