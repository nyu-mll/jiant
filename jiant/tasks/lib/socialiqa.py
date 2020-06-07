from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines, read_file_lines


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return SocialIQATask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class SocialIQATask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = [1, 2, 3]
    CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
    NUM_CHOICES = len(CHOICE_KEYS)

    def get_train_examples(self):
        return self._create_examples(
            lines=read_json_lines(self.train_path),
            labels=self._read_labels(self.path_dict["train_labels"]),
            set_type="train",
        )

    def get_val_examples(self):
        return self._create_examples(
            lines=read_json_lines(self.val_path),
            labels=self._read_labels(self.path_dict["val_labels"]),
            set_type="val",
        )

    def get_test_examples(self):
        raise NotImplementedError("get_test_examples")

    @classmethod
    def _create_examples(cls, lines, labels, set_type):
        examples = []
        answer_key_ls = ["answerA", "answerB", "answerC"]
        for i, (line, label) in enumerate(zip(lines, labels)):
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    prompt=line["context"] + " " + line["question"],
                    choice_list=[line[answer_key] for answer_key in answer_key_ls],
                    label=label,
                )
            )
        return examples

    @classmethod
    def _read_labels(cls, path):
        lines = read_file_lines(path)
        return [int(line.strip()) for line in lines]
