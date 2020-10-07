from dataclasses import dataclass

from jiant.utils.python.datastructures import ReusableGenerator
from jiant.tasks.lib.templates import mlm as mlm_template


@dataclass
class Example(mlm_template.Example):
    pass


@dataclass
class TokenizedExample(mlm_template.TokenizedExample):
    pass


@dataclass
class DataRow(mlm_template.DataRow):
    pass


@dataclass
class Batch(mlm_template.Batch):
    pass


@dataclass
class MaskedBatch(mlm_template.MaskedBatch):
    pass


class MLMSimpleTask(mlm_template.MLMTask):
    """Simple implementation of MLM task
    - Reads from a single file per phase
    - One example per line (examples that are too long will be truncated)
    - Empty lines are skipped.
    """

    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    def __init__(self, name, path_dict, mlm_probability=0.15, do_mask=True):
        super().__init__(name=name, path_dict=path_dict)
        self.mlm_probability = mlm_probability
        self.do_mask = do_mask

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train", return_generator=True)

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val", return_generator=True)

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test", return_generator=True)

    @classmethod
    def _get_examples_generator(cls, path, set_type):
        with open(path, "r") as f:
            for (i, line) in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield Example(
                    guid="%s-%s" % (set_type, i), text=line,
                )

    @classmethod
    def _create_examples(cls, path, set_type, return_generator):
        generator = ReusableGenerator(cls._get_examples_generator, path=path, set_type=set_type)
        if return_generator:
            return generator
        else:
            return list(generator)
