from dataclasses import dataclass

from jiant.shared.constants import PHASE
from jiant.tasks.lib.templates.squad_style import core as squad_style_template
from jiant.utils.python.io import read_jsonl


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


class NewsQATask(squad_style_template.BaseSquadStyleTask):
    Example = Example
    DataRow = DataRow
    Batch = Batch

    def get_train_examples(self):
        return self.read_examples(path=self.train_path, set_type=PHASE.TRAIN)

    def get_val_examples(self):
        return self.read_examples(path=self.val_path, set_type=PHASE.VAL)

    @classmethod
    def read_examples(cls, path: str, set_type: str):
        examples = []
        for entry in read_jsonl(path):
            for qa in entry["qas"]:
                answer_text = entry["text"][qa["answer"]["s"] : qa["answer"]["e"]]
                examples.append(
                    Example(
                        qas_id=f"{set_type}-{len(examples)}",
                        question_text=qa["question"],
                        context_text=entry["text"],
                        answer_text=answer_text,
                        start_position_character=qa["answer"]["s"],
                        title="",
                        is_impossible=False,
                        answers=[{"answer_start": qa["answer"]["s"], "text": answer_text}],
                    )
                )
        return examples
