from dataclasses import dataclass

from . import mnli


@dataclass
class Example(mnli.Example):
    pass


@dataclass
class TokenizedExample(mnli.TokenizedExample):
    pass


@dataclass
class DataRow(mnli.DataRow):
    pass


@dataclass
class Batch(mnli.Batch):
    pass


class MnliMismatchedTask(mnli.MnliTask):
    def get_train_examples(self):
        raise RuntimeError("This task does not support training examples")
