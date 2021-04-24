from dataclasses import dataclass
from . import base as base
from jiant.tasks.lib.templates.shared import labels_to_bimap


@dataclass
class Example(base.Example):
    @property
    def label_to_id(self):
        return SentEvalTopConstituentsTask.LABEL_TO_ID


@dataclass
class TokenizedExample(base.TokenizedExample):
    pass


@dataclass
class DataRow(base.DataRow):
    pass


@dataclass
class Batch(base.Batch):
    pass


class SentEvalTopConstituentsTask(base.BaseSentEvalTask):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    LABELS = [
        "ADVP_NP_VP_.",
        "CC_ADVP_NP_VP_.",
        "CC_NP_VP_.",
        "IN_NP_VP_.",
        "NP_ADVP_VP_.",
        "NP_NP_VP_.",
        "NP_PP_.",
        "NP_VP_.",
        "OTHER",
        "PP_NP_VP_.",
        "RB_NP_VP_.",
        "SBAR_NP_VP_.",
        "SBAR_VP_.",
        "S_CC_S_.",
        "S_NP_VP_.",
        "S_VP_.",
        "VBD_NP_VP_.",
        "VP_.",
        "WHADVP_SQ_.",
        "WHNP_SQ_.",
    ]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)
