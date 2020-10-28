from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import multiple_choice as mc_template
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(mc_template.Example):
    @property
    def task(self):
        return CommonsenseQATask


@dataclass
class TokenizedExample(mc_template.TokenizedExample):
    pass


@dataclass
class DataRow(mc_template.DataRow):
    pass


@dataclass
class Batch(mc_template.Batch):
    pass


class CommonsenseQATask(mc_template.AbstractMultipleChoiceTask):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    CHOICE_KEYS = ["A", "B", "C", "D", "E"]
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
            examples.append(cls._create_example(raw_example=line, set_type=set_type, i=i,))
        return examples

    @classmethod
    def _create_example(cls, raw_example, set_type, i):
        # Use heuristic for determining original or HF Datasets format
        if isinstance(raw_example["question"], dict):
            return cls._create_example_from_original_format(
                raw_example=raw_example, set_type=set_type, i=i,
            )
        elif isinstance(raw_example["question"], str):
            return cls._create_example_from_hf_datasets_format(
                raw_example=raw_example, set_type=set_type, i=i,
            )
        else:
            raise TypeError(raw_example["question"])

    @classmethod
    def _create_example_from_original_format(cls, raw_example, set_type, i):
        """Return question and choices from original example format"""
        choice_dict = {elem["label"]: elem["text"] for elem in raw_example["question"]["choices"]}
        choice_list = [choice_dict[key] for key in cls.CHOICE_KEYS]
        return Example(
            guid="%s-%s" % (set_type, i),
            prompt=raw_example["question"]["stem"],
            choice_list=choice_list,
            label=raw_example["answerKey"] if set_type != "test" else cls.CHOICE_KEYS[-1],
        )

    @classmethod
    def _create_example_from_hf_datasets_format(cls, raw_example, set_type, i):
        """Return question and choices from HF Datasets example format"""
        return Example(
            guid="%s-%s" % (set_type, i),
            prompt=raw_example["question"],
            choice_list=raw_example["choices"]["text"],
            label=raw_example["answerKey"] if set_type != "test" else cls.CHOICE_KEYS[-1],
        )
