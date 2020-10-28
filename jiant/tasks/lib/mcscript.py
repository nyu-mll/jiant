from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return MCScriptTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class MCScriptTask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = [0, 1]
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
        for line in lines:
            passage = line["passage"]["text"]
            passage_id = line["idx"]
            for question_dict in line["passage"]["questions"]:
                question = question_dict["question"]
                question_id = question_dict["idx"]
                answer_dicts = question_dict["answers"]
                examples.append(
                    Example(
                        guid="%s-%s-%s" % (set_type, passage_id, question_id),
                        prompt=passage,
                        choice_list=[
                            question + " " + answer_dict["text"] for answer_dict in answer_dicts
                        ],
                        label=answer_dicts[1]["label"] == "True"
                        if set_type != "test"
                        else cls.CHOICE_KEYS[-1],
                    )
                )

        return examples
