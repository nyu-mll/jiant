import pandas as pd
from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return CosmosQATask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class CosmosQATask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = [0, 1, 2, 3]
    CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
    NUM_CHOICES = len(CHOICE_KEYS)

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".jsonl"):
            df = pd.read_json(path, lines=True)
        else:
            raise RuntimeError("Format not supported")
        examples = []
        for i, row in enumerate(df.itertuples()):
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    prompt=row.context + " " + row.question,
                    choice_list=[row.answer0, row.answer1, row.answer2, row.answer3],
                    label=row.label if set_type != "test" else cls.CHOICE_KEYS[-1],
                )
            )
        return examples
