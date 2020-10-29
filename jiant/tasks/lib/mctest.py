from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_file_lines


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return MCTestTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class MCTestTask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = ["A", "B", "C", "D"]
    CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
    NUM_CHOICES = len(CHOICE_KEYS)

    def get_train_examples(self):
        return self._create_examples(
            lines=read_file_lines(self.train_path, strip_lines=True),
            ans_lines=read_file_lines(self.path_dict["train_ans"], strip_lines=True),
            set_type="train",
        )

    def get_val_examples(self):
        return self._create_examples(
            lines=read_file_lines(self.val_path, strip_lines=True),
            ans_lines=read_file_lines(self.path_dict["val_ans"], strip_lines=True),
            set_type="val",
        )

    def get_test_examples(self):
        return self._create_examples(
            lines=read_file_lines(self.test_path, strip_lines=True),
            ans_lines=None,
            set_type="test",
        )

    @classmethod
    def _create_examples(cls, lines, ans_lines, set_type):
        examples = []
        if ans_lines is None:
            ans_lines = ["\t".join([cls.CHOICE_KEYS[-1]] * 4) for line in lines]
        for i, (line, ans) in enumerate(zip(lines, ans_lines)):
            line = line.split("\t")
            ans = ans.split("\t")
            for j in range(4):
                examples.append(
                    Example(
                        guid="%s-%s" % (set_type, i * 4 + j),
                        prompt=line[2].replace("\\newline", " ") + " " + line[3 + j * 5],
                        choice_list=line[4 + j * 5 : 8 + j * 5],
                        label=ans[j],
                    )
                )
        return examples
