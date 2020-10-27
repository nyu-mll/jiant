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


class QuorefTask(squad_style_template.BaseSquadStyleTask):
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
        for i, line in enumerate(read_jsonl(path)):
            if set_type == PHASE.TRAIN:
                for j, (answer_start, answer_text) in enumerate(
                    zip(line["answers"]["answer_start"], line["answers"]["text"])
                ):
                    examples.append(
                        Example(
                            qas_id=f"{set_type}-{i}",
                            question_text=line["question"],
                            context_text=line["context"],
                            answer_text=answer_text,
                            start_position_character=answer_start,
                            title=line["title"],
                            is_impossible=False,
                            answers=[],
                        )
                    )
            else:
                answers = [
                    {"answer_start": answer_start, "text": answer_text}
                    for answer_start, answer_text in zip(
                        line["answers"]["answer_start"], line["answers"]["text"]
                    )
                ]
                examples.append(
                    Example(
                        qas_id=f"{set_type}-{i}",
                        question_text=line["question"],
                        context_text=line["context"],
                        answer_text=None,
                        start_position_character=None,
                        title=line["title"],
                        is_impossible=False,
                        answers=answers,
                    )
                )
        return examples
