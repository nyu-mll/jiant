from dataclasses import dataclass

from jiant.tasks.lib.templates.squad_style import core as squad_style_template


@dataclass
class Example(squad_style_template.Example):
    def tokenize(self, tokenizer):
        raise NotImplementedError("SQuaD is weird")


@dataclass
class DataRow(squad_style_template.DataRow):
    pass


@dataclass
class Batch(squad_style_template.Batch):
    pass


class SquadTask(squad_style_template.BaseSquadStyleTask):
    Example = Example
    DataRow = DataRow
    Batch = Batch
