from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines, read_file_lines


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return PiqaTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class PiqaTask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = [0, 1]
    CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
    NUM_CHOICES = len(CHOICE_KEYS)

    def get_train_examples(self):
        return self._create_examples(
            lines=zip(
                read_json_lines(self.train_path),
                read_file_lines(self.path_dict["train_labels"], strip_lines=True),
            ),
            set_type="train",
        )

    def get_val_examples(self):
        return self._create_examples(
            lines=zip(
                read_json_lines(self.val_path),
                read_file_lines(self.path_dict["val_labels"], strip_lines=True),
            ),
            set_type="val",
        )

    def get_test_examples(self):
        return self._create_examples(
            lines=zip(read_json_lines(self.test_path), read_json_lines(self.test_path)),
            set_type="test",
        )

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []

        for i, (ex, label_string) in enumerate(lines):
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    prompt=ex["goal"],
                    choice_list=[ex["sol1"], ex["sol2"]],
                    label=int(label_string) if set_type != "test" else cls.CHOICE_KEYS[-1],
                )
            )

        return examples
