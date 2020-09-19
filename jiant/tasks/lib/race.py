import glob
import os

from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return RaceTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class RaceTask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = ["A", "B", "C", "D"]
    CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
    NUM_CHOICES = len(CHOICE_KEYS)

    def get_train_examples(self):
        return self._create_examples(self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(self.test_path, set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        examples = []
        for i, path in enumerate(sorted(glob.glob(os.path.join(path, "*.txt")))):
            raw_example = read_json(path)
            for qn_i, question in enumerate(raw_example["questions"]):
                examples.append(
                    Example(
                        guid="%s-%s" % (set_type, i),
                        prompt=raw_example["article"],
                        choice_list=raw_example["options"][qn_i],
                        label=raw_example["answers"][qn_i] if set_type != "test" else cls.CHOICE_KEYS[-1],
                    )
                )
        return examples
