from dataclasses import dataclass
import jiant.tasks.lib.acceptability_judgement.base as base


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


class AcceptabilityEOSTask(base.BaseAcceptabilityTask):
    pass
