from dataclasses import dataclass
from . import base as base
from jiant.tasks.lib.templates.shared import labels_to_bimap


@dataclass
class Example(base.Example):
    @property
    def label_to_id(self):
        return SentEvalSentenceLengthTask.LABEL_TO_ID


@dataclass
class TokenizedExample(base.TokenizedExample):
    pass


@dataclass
class DataRow(base.DataRow):
    pass


@dataclass
class Batch(base.Batch):
    pass


class SentEvalSentenceLengthTask(base.BaseSentEvalTask):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    LABELS = [0, 1, 2, 3, 4, 5, 6]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)
