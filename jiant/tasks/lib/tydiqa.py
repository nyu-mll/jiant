from dataclasses import dataclass

from jiant.tasks.lib.templates.squad_style import core as squad_style_template


@dataclass
class Example(squad_style_template.Example):
    def tokenize(self, tokenizer):
        raise NotImplementedError("SQuaD is weird")


@dataclass
class DataRow(squad_style_template.DataRow):
    pass


@dataclass
class Batch(squad_style_template.Batch):
    pass


class TyDiQATask(squad_style_template.BaseSquadStyleTask):
    Example = Example
    DataRow = DataRow
    Batch = Batch

    def __init__(
        self,
        name,
        path_dict,
        language,
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
    ):
        super().__init__(
            name=name,
            path_dict=path_dict,
            version_2_with_negative=version_2_with_negative,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            null_score_diff_threshold=null_score_diff_threshold,
        )
        self.language = language

    def get_train_examples(self):
        if self.language == "en":
            return self.read_squad_examples(path=self.train_path, set_type="train")
        else:
            raise NotImplementedError("TyDiQA does not have training examples except for English")

    @classmethod
    def read_squad_examples(cls, path, set_type):
        return squad_style_template.generic_read_squad_examples(
            path=path, set_type=set_type, example_class=cls.Example,
        )
