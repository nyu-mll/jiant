from dataclasses import dataclass

import pandas as pd

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines
from typing import List

@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return ArctTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class WinograndeTask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = [0, 1]
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
        df_names = [
            "#id",
            "warrant0",
            "warrant1",
            "gold_label",
            "reason",
            "claim",
            "debateTitle",
            "debateInfo",
        ]

        df = pd.read_csv(path, sep="\t", header=True,names=df_names,)
        choice_pre = "And since "
        examples = []

        for i, row in enumerate(df.itertuples()):
            # Repo explanation from https://github.com/UKPLab/argument-reasoning-comprehension-task
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    prompt=row.reason+' ',
                    choice_list=[choice_pre + row.warrant0 + ', ' + row.claim,
                                 choice_pre + row.warrant1 + ', ' + row.claim],
                    label=row.gold_label if set_type != "test" else cls.CHOICE_KEYS[-1],
                )
            )

            # BERT input from https://arxiv.org/pdf/1907.07355.pdf
            # examples.append(
            #     Example(
            #         guid="%s-%s" % (set_type, i),
            #         prompt=row.claim + ' ' + row.reason,
            #         choice_list=[row.warrant0,
            #                      row.warrant1],
            #         label=row.gold_label if set_type != "test" else cls.CHOICE_KEYS[-1],
            #     )
            # )

        return examples
