import gzip
import json
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


class MrqaNaturalQuestionsTask(squad_style_template.BaseSquadStyleTask):
    Example = Example
    DataRow = DataRow
    Batch = Batch

    def get_train_examples(self):
        return self.read_examples(path=self.train_path, set_type=PHASE.TRAIN)

    def get_val_examples(self):
        return self.read_examples(path=self.val_path, set_type=PHASE.VAL)

    @classmethod
    def read_examples(cls, path: str, set_type: str):
        if path.endswith(".gz"):
            with gzip.open(path, "r") as f:
                data = [json.loads(row) for row in f.readlines()]
        elif path.endswith(".jsonl"):
            data = read_jsonl(path)
        else:
            raise KeyError(f"Unknown format: {path}")

        # First row is a header row
        assert "header" in data[0]

        examples = []
        for i, line in enumerate(data[1:]):
            for elem in line["qas"]:
                if set_type == PHASE.TRAIN:
                    # Just use first answer for training (e.g. "Car", "Vehicle")
                    answer = elem["detected_answers"][0]
                    # Just use first occurrence of answer for training ("Every car is a car.")
                    answer_span = answer["char_spans"][0]
                    answer_text = line["context"][answer_span[0] : answer_span[1] + 1]
                    # assert len(elem["detected_answers"][0]["char_spans"]) == 1
                    examples.append(
                        Example(
                            qas_id=f"{set_type}-{i}",
                            question_text=elem["question"],
                            context_text=line["context"],
                            answer_text=answer_text,
                            start_position_character=answer_span[0],
                            title="",
                            is_impossible=False,
                            answers=[],
                        )
                    )
                else:
                    answers = []
                    for answer in elem["detected_answers"]:
                        for answer_span in answer["char_spans"]:
                            answers.append(
                                {
                                    "answer_start": answer_span[0],
                                    "text": line["context"][answer_span[0] : answer_span[1] + 1],
                                }
                            )
                    examples.append(
                        Example(
                            qas_id=f"{set_type}-{i}",
                            question_text=elem["question"],
                            context_text=line["context"],
                            answer_text=None,
                            start_position_character=None,
                            title="",
                            is_impossible=False,
                            answers=answers,
                        )
                    )
        return examples
