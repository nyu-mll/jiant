from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return QuailTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class QuailTask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = ["0", "1", "2", "3"]
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
        examples = []
        for i, line in enumerate(lines):
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    prompt=line["context"],
                    choice_list=[d for d in line["answers"]],
                    label=line["label"],
                )
            )
        return examples
