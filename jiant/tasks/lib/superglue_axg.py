from dataclasses import dataclass

from . import rte


@dataclass
class Example(rte.Example):
    pass


@dataclass
class TokenizedExample(rte.Example):
    pass


@dataclass
class DataRow(rte.DataRow):
    pass


@dataclass
class Batch(rte.Batch):
    pass


class SuperglueWinogenderDiagnosticsTask(rte.RteTask):
    def get_train_examples(self):
        raise RuntimeError("This task does not support training examples")

    def get_val_examples(self):
        raise RuntimeError("This task does not support validation examples")
