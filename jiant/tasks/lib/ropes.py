from dataclasses import dataclass
from typing import Optional

from jiant.shared.constants import PHASE
from jiant.tasks.lib.templates.squad_style import core as squad_style_template
from jiant.utils.python.io import read_json
from jiant.utils.display import maybe_tqdm


@dataclass
class Example(squad_style_template.Example):
    # Additional fields
    background_text: Optional[str] = None
    situation_text: Optional[str] = None

    def tokenize(self, tokenizer):
        raise NotImplementedError("SQuaD is weird")


@dataclass
class DataRow(squad_style_template.DataRow):
    pass


@dataclass
class Batch(squad_style_template.Batch):
    pass


class RopesTask(squad_style_template.BaseSquadStyleTask):
    Example = Example
    DataRow = DataRow
    Batch = Batch

    def get_train_examples(self):
        return self.read_examples(path=self.train_path, set_type=PHASE.TRAIN)

    def get_val_examples(self):
        return self.read_examples(path=self.val_path, set_type=PHASE.VAL)

    def get_test_examples(self):
        return self.read_examples(path=self.test_path, set_type=PHASE.TEST)

    @classmethod
    def read_examples(cls, path, set_type):
        input_data = read_json(path, encoding="utf-8")["data"]

        is_training = set_type == PHASE.TRAIN
        examples = []
        for entry in maybe_tqdm(input_data, desc="Reading SQuAD Entries"):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = Example(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=paragraph["background"] + " " + paragraph["situation"],
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                        background_text=paragraph["background"],
                        situation_text=paragraph["situation"],
                    )
                    examples.append(example)
        return examples
