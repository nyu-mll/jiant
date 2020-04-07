import json
import logging as log
import os

from jiant.tasks.tasks import PairClassificationTask
from jiant.tasks.registry import register_task  # global task registry

from jiant.utils.data_loaders import tokenize_and_truncate


class BaseXNLITask(PairClassificationTask):
    """ Base Task class for XNLI """

    LANGUAGE = None

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, n_classes=3, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def _load_jsonl(self, data_file):
        targ_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
        data = [json.loads(d) for d in open(data_file, encoding="utf-8")]
        sent1s, sent2s, trgs, idxs = [], [], [], []
        for example in data:
            if not example["language"] == self.LANGUAGE:
                continue
            sent1s.append(
                tokenize_and_truncate(self._tokenizer_name, example["sentence1"], self.max_seq_len)
            )
            sent2s.append(
                tokenize_and_truncate(self._tokenizer_name, example["sentence2"], self.max_seq_len)
            )
            trg = targ_map[example["gold_label"]] if "label" in example else 0
            trgs.append(trg)
            idxs.append(len(idxs))
        return [sent1s, sent2s, trgs, idxs]

    def load_data(self):
        """ Process the datasets located at path. """
        self.val_data_text = self._load_jsonl(os.path.join(self.path, "xnli.dev.jsonl"))
        self.test_data_text = self._load_jsonl(os.path.join(self.path, "xnli.test.jsonl"))

        self.sentences = self.val_data_text[0] + self.val_data_text[1]
        log.info(f"\tFinished loading XNLI ({self.LANGUAGE})")


@register_task("xnli_ar", rel_path="XNLI/")
class XNLIArTask(BaseXNLITask):
    LANGUAGE = "ar"


@register_task("xnli_bg", rel_path="XNLI/")
class XNLIBgTask(BaseXNLITask):
    LANGUAGE = "bg"


@register_task("xnli_de", rel_path="XNLI/")
class XNLIDeTask(BaseXNLITask):
    LANGUAGE = "de"


@register_task("xnli_el", rel_path="XNLI/")
class XNLIElTask(BaseXNLITask):
    LANGUAGE = "el"


@register_task("xnli_en", rel_path="XNLI/")
class XNLIEnTask(BaseXNLITask):
    LANGUAGE = "en"


@register_task("xnli_es", rel_path="XNLI/")
class XNLIEsTask(BaseXNLITask):
    LANGUAGE = "es"


@register_task("xnli_fr", rel_path="XNLI/")
class XNLIFrTask(BaseXNLITask):
    LANGUAGE = "fr"


@register_task("xnli_hi", rel_path="XNLI/")
class XNLIHiTask(BaseXNLITask):
    LANGUAGE = "hi"


@register_task("xnli_ru", rel_path="XNLI/")
class XNLIRuTask(BaseXNLITask):
    LANGUAGE = "ru"


@register_task("xnli_sw", rel_path="XNLI/")
class XNLISwTask(BaseXNLITask):
    LANGUAGE = "sw"


@register_task("xnli_th", rel_path="XNLI/")
class XNLIThTask(BaseXNLITask):
    LANGUAGE = "th"


@register_task("xnli_tr", rel_path="XNLI/")
class XNLITrTask(BaseXNLITask):
    LANGUAGE = "tr"


@register_task("xnli_ur", rel_path="XNLI/")
class XNLIUrTask(BaseXNLITask):
    LANGUAGE = "ur"


@register_task("xnli_vi", rel_path="XNLI/")
class XNLIViTask(BaseXNLITask):
    LANGUAGE = "vi"


@register_task("xnli_zh", rel_path="XNLI/")
class XNLIZhTask(BaseXNLITask):
    LANGUAGE = "zh"
