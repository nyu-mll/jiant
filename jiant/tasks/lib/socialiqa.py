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

    CHOICE_KEYS = ["A", "B", "C"]
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
        answer_key_ls = ["answerA", "answerB", "answerC"]
        hf_datasets_label_map = {
            "1\n": "A",
            "2\n": "B",
            "3\n": "C",
        }
        for i, line in enumerate(lines):
            if "label" in line:
                # Loading from HF Datasets data
                label = hf_datasets_label_map[line["label"]]
            else:
                # Loading from original data
                label = line["correct"]
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
