from dataclasses import dataclass
from . import base as base


@dataclass
class Example(base.Example):
    pass


@dataclass
class TokenizedExample(base.TokenizedExample):
    pass


@dataclass
class DataRow(base.DataRow):
    pass


@dataclass
class Batch(base.Batch):
    pass


class AcceptabilityCoordTask(base.BaseAcceptabilityTask):
    pass
