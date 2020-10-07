import string
import re
import sys
import unicodedata
from collections import Counter

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


class MlqaTask(squad_style_template.BaseSquadStyleTask):
    Example = Example
    DataRow = DataRow
    Batch = Batch

    def __init__(
        self,
        name,
        path_dict,
        context_language,
        question_language,
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
        self.context_language = context_language
        self.question_language = question_language

    def get_train_examples(self):
        raise NotImplementedError("MLQA does not have training examples")

    @classmethod
    def read_squad_examples(cls, path, set_type):
        return squad_style_template.generic_read_squad_examples(
            path=path, set_type=set_type, example_class=cls.Example,
        )


# === Evaluation === #
# MLQA has a slightly different evaluation / detokenization for different languages.
# Can de-dup this later if necessary.

PUNCT = {
    chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
}.union(string.punctuation)
WHITESPACE_LANGS = ["en", "es", "hi", "vi", "de", "ar"]
MIXED_SEGMENTATION_LANGS = ["zh"]


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r"[\u4e00-\u9fa5]", char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, inner_lang):
        if inner_lang == "en":
            return re.sub(r"\b(a|an|the)\b", " ", text)
        elif inner_lang == "es":
            return re.sub(r"\b(un|una|unos|unas|el|la|los|las)\b", " ", text)
        elif inner_lang == "hi":
            return text  # Hindi does not have formal articles
        elif inner_lang == "vi":
            return re.sub(r"\b(của|là|cái|chiếc|những)\b", " ", text)
        elif inner_lang == "de":
            return re.sub(
                r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b", " ", text
            )
        elif inner_lang == "ar":
            return re.sub("\sال^|ال", " ", text)  # noqa: W605
        elif inner_lang == "zh":
            return text  # Chinese does not have formal articles
        else:
            raise Exception("Unknown Language {}".format(inner_lang))

    def white_space_fix(text, inner_lang):
        if inner_lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif inner_lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception("Unknown Language {}".format(inner_lang))
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)


def f1_score(prediction, ground_truth, lang):
    prediction_tokens = normalize_answer(prediction, lang).split()
    ground_truth_tokens = normalize_answer(ground_truth, lang).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, lang):
    return normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, lang):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, lang)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, lang):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths, lang
                )
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths, lang)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}
