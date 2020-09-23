"""Semantic role labeling Edge Probing task.

Task source paper: https://arxiv.org/pdf/1905.06316.pdf.
Task data prep directions: https://github.com/nyu-mll/jiant/blob/master/probing/data/README.md.

"""
from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import edge_probing_two_span
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(edge_probing_two_span.Example):
    @property
    def task(self):
        return SrlTask


@dataclass
class TokenizedExample(edge_probing_two_span.TokenizedExample):
    pass


@dataclass
class DataRow(edge_probing_two_span.DataRow):
    pass


@dataclass
class Batch(edge_probing_two_span.Batch):
    pass


class SrlTask(edge_probing_two_span.AbstractProbingTask):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    LABELS = [
        "ARG0",
        "ARG1",
        "ARG2",
        "ARG3",
        "ARG4",
        "ARG5",
        "ARGA",
        "ARGM-ADJ",
        "ARGM-ADV",
        "ARGM-CAU",
        "ARGM-COM",
        "ARGM-DIR",
        "ARGM-DIS",
        "ARGM-DSP",
        "ARGM-EXT",
        "ARGM-GOL",
        "ARGM-LOC",
        "ARGM-LVB",
        "ARGM-MNR",
        "ARGM-MOD",
        "ARGM-NEG",
        "ARGM-PNC",
        "ARGM-PRD",
        "ARGM-PRP",
        "ARGM-PRR",
        "ARGM-PRX",
        "ARGM-REC",
        "ARGM-TMP",
        "C-ARG0",
        "C-ARG1",
        "C-ARG2",
        "C-ARG3",
        "C-ARG4",
        "C-ARGM-ADJ",
        "C-ARGM-ADV",
        "C-ARGM-CAU",
        "C-ARGM-COM",
        "C-ARGM-DIR",
        "C-ARGM-DIS",
        "C-ARGM-DSP",
        "C-ARGM-EXT",
        "C-ARGM-LOC",
        "C-ARGM-MNR",
        "C-ARGM-MOD",
        "C-ARGM-NEG",
        "C-ARGM-PRP",
        "C-ARGM-TMP",
        "R-ARG0",
        "R-ARG1",
        "R-ARG2",
        "R-ARG3",
        "R-ARG4",
        "R-ARG5",
        "R-ARGM-ADV",
        "R-ARGM-CAU",
        "R-ARGM-COM",
        "R-ARGM-DIR",
        "R-ARGM-EXT",
        "R-ARGM-GOL",
        "R-ARGM-LOC",
        "R-ARGM-MNR",
        "R-ARGM-MOD",
        "R-ARGM-PNC",
        "R-ARGM-PRD",
        "R-ARGM-PRP",
        "R-ARGM-TMP",
    ]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    @property
    def num_spans(self):
        return 2

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (line_num, line) in enumerate(lines):
            for (target_num, target) in enumerate(line["targets"]):
                span1 = target["span1"]
                span2 = target["span2"]
                examples.append(
                    Example(
                        guid="%s-%s-%s" % (set_type, line_num, target_num),
                        text=line["text"],
                        span1=span1,
                        span2=span2,
                        labels=[target["label"]] if set_type != "test" else [cls.LABELS[-1]],
                    )
                )
        return examples
