from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines
from jiant.tasks.core import SuperGlueMixin


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return CopaTask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class CopaTask(SuperGlueMixin, mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = [0, 1]
    CHOICE_TO_ID, ID_TO_CHOICE = labels_to_bimap(CHOICE_KEYS)
    NUM_CHOICES = len(CHOICE_KEYS)

    _QUESTION_DICT = {
        "cause": "What was the cause of this?",
        "effect": "What happened as a result?",
    }

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
            question = cls._QUESTION_DICT[line["question"]]
            examples.append(
                Example(
                    # NOTE: CopaTask.super_glue_format_preds() is dependent on this guid format.
                    guid="%s-%s" % (set_type, line["idx"]),
                    prompt=line["premise"] + " " + question,
                    choice_list=[line["choice1"], line["choice2"]],
                    label=line["label"] if set_type != "test" else cls.CHOICE_KEYS[-1],
                )
            )
        return examples

    @classmethod
    def super_glue_format_preds(cls, pred_dict):
        """Reformat this task's raw predictions to have the structure expected by SuperGLUE."""
        lines = []
        for pred, guid in zip(list(pred_dict["preds"]), list(pred_dict["guids"])):
            lines.append({"idx": int(guid.split("-")[1]), "label": cls.CHOICE_KEYS[pred]})
        return lines
