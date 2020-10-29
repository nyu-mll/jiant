from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return ArcChallengeTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class ArcChallengeTask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = ["A", "B", "C", "D", "E"]
    CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
    NUM_CHOICES = len(CHOICE_KEYS)

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        potential_label_map = {
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
            "5": "E",
        }
        NUM_CHOICES = len(potential_label_map)
        examples = []
        for i, line in enumerate(lines):
            label = line["answerKey"]
            if label in potential_label_map:
                label = potential_label_map[label]
            choice_list = [d for d in line["choices"]["text"]]
            filler_choice_list = ["." for i in range(NUM_CHOICES - len(choice_list))]
            choice_list = choice_list + filler_choice_list
            assert len(choice_list) == NUM_CHOICES

            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    prompt=line["question"],
                    choice_list=choice_list,
                    label=label,
                )
            )
        return examples
