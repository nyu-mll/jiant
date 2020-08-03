"""Relation Classification Edge Probing task.

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
        return SemevalTask


@dataclass
class TokenizedExample(edge_probing_two_span.TokenizedExample):
    pass


@dataclass
class DataRow(edge_probing_two_span.DataRow):
    pass


@dataclass
class Batch(edge_probing_two_span.Batch):
    pass


class SemevalTask(edge_probing_two_span.AbstractProbingTask):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    LABELS = [
        "Cause-Effect(e1,e2)",
        "Cause-Effect(e2,e1)",
        "Component-Whole(e1,e2)",
        "Component-Whole(e2,e1)",
        "Content-Container(e1,e2)",
        "Content-Container(e2,e1)",
        "Entity-Destination(e1,e2)",
        "Entity-Destination(e2,e1)",
        "Entity-Origin(e1,e2)",
        "Entity-Origin(e2,e1)",
        "Instrument-Agency(e1,e2)",
        "Instrument-Agency(e2,e1)",
        "Member-Collection(e1,e2)",
        "Member-Collection(e2,e1)",
        "Message-Topic(e1,e2)",
        "Message-Topic(e2,e1)",
        "Other",
        "Product-Producer(e1,e2)",
        "Product-Producer(e2,e1)",
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
