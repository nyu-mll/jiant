"""Task definitions for question answering tasks."""
import os
import re
import json
import string
import collections
import gzip
import random
from typing import Iterable, Sequence, Type

import torch
import logging as log
from allennlp.training.metrics import Average, F1Measure, CategoricalAccuracy
from allennlp.data.fields import LabelField, MetadataField
from allennlp.data import Instance
from jiant.allennlp_mods.numeric_field import NumericField
from jiant.allennlp_mods.span_metrics import SpanF1Measure

from jiant.utils.data_loaders import tokenize_and_truncate

from jiant.tasks.tasks import Task, SpanPredictionTask, MultipleChoiceTask
from jiant.tasks.tasks import sentence_to_text_field
from jiant.tasks.registry import register_task
from ..utils.retokenize import get_aligner_fn


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.
    From official ReCoRD eval script """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """ Compute normalized token level F1
    From official ReCoRD eval script """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """ Compute normalized exact match
    From official ReCoRD eval script """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """ Compute max metric between prediction and each ground truth.
    From official ReCoRD eval script """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


@register_task("multirc", rel_path="MultiRC/")
class MultiRCTask(Task):
    """Multi-sentence Reading Comprehension task
    See paper at https://cogcomp.org/multirc/ """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, **kw)
        self.scorer1 = F1Measure(positive_label=1)
        self.scorer2 = Average()  # to delete
        self.scorer3 = F1Measure(positive_label=1)
        self._score_tracker = collections.defaultdict(list)
        self.val_metric = "%s_avg" % self.name
        self.val_metric_decreases = False
        self.max_seq_len = max_seq_len
        self.files_by_split = {
            "train": os.path.join(path, "train.jsonl"),
            "val": os.path.join(path, "val.jsonl"),
            "test": os.path.join(path, "test.jsonl"),
        }

    def load_data(self):
        # Data is exposed as iterable: no preloading
        pass

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.

        Split should be one of "train", "val", or "test".
        """
        return self.load_data_for_path(self.files_by_split[split])

    def load_data_for_path(self, path):
        """ Load data """

        with open(path, encoding="utf-8") as data_fh:
            examples = []
            for example in data_fh:
                ex = json.loads(example)

                assert (
                    "version" in ex and ex["version"] == 1.1
                ), "MultiRC version is invalid! Example indices are likely incorrect. "
                "Please re-download the data from super.gluebenchmark.com ."

                # each example has a passage field -> (text, questions)
                # text is the passage, which requires some preprocessing
                # questions is a list of questions, has fields (question, sentences_used, answers)
                ex["passage"]["text"] = tokenize_and_truncate(
                    self.tokenizer_name, ex["passage"]["text"], self.max_seq_len
                )
                for question in ex["passage"]["questions"]:
                    question["question"] = tokenize_and_truncate(
                        self.tokenizer_name, question["question"], self.max_seq_len
                    )
                    for answer in question["answers"]:
                        answer["text"] = tokenize_and_truncate(
                            self.tokenizer_name, answer["text"], self.max_seq_len
                        )
                examples.append(ex)
        return examples

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for example in self.load_data_for_path(path):
                yield example["passage"]["text"]
                for question in example["passage"]["questions"]:
                    yield question["question"]
                    for answer in question["answers"]:
                        yield answer["text"]

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(passage, question, answer, label, par_idx, qst_idx, ans_idx):
            """ pq_id: passage-question ID """
            d = {}
            d["psg_str"] = MetadataField(" ".join(passage))
            d["qst_str"] = MetadataField(" ".join(question))
            d["ans_str"] = MetadataField(" ".join(answer))
            d["psg_idx"] = MetadataField(par_idx)
            d["qst_idx"] = MetadataField(qst_idx)
            d["ans_idx"] = MetadataField(ans_idx)
            d["idx"] = MetadataField(ans_idx)  # required by evaluate()
            if model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                inp = model_preprocessing_interface.boundary_token_fn(para, question + answer)
                d["psg_qst_ans"] = sentence_to_text_field(inp, indexers)
            else:
                d["psg"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(passage), indexers
                )
                d["qst"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(question), indexers
                )
                d["ans"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(answer), indexers
                )
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)

            return Instance(d)

        for example in split:
            par_idx = example["idx"]
            para = example["passage"]["text"]
            for ex in example["passage"]["questions"]:
                qst_idx = ex["idx"]
                question = ex["question"]
                for answer in ex["answers"]:
                    ans_idx = answer["idx"]
                    ans = answer["text"]
                    label = int(answer["label"]) if "label" in answer else 0
                    yield _make_instance(para, question, ans, label, par_idx, qst_idx, ans_idx)

    def count_examples(self):
        """ Compute here b/c we"re streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(
                len(q["answers"])
                for r in open(split_path, "r", encoding="utf-8")
                for q in json.loads(r)["passage"]["questions"]
            )

        self.example_counts = example_counts

    def update_metrics(self, logits, labels, idxs, tagmask=None):
        """ A batch of logits, labels, and the passage+questions they go with """
        self.scorer1(logits, labels)
        logits, labels = logits.detach().cpu(), labels.detach().cpu()
        # track progress on each question
        for ex, logit, label in zip(idxs, logits, labels):
            self._score_tracker[ex].append((logit, label))

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        _, _, ans_f1 = self.scorer1.get_metric(reset)

        ems, f1s = [], []
        for logits_and_labels in self._score_tracker.values():
            logits, labels = list(zip(*logits_and_labels))
            logits = torch.stack(logits)
            labels = torch.stack(labels)

            # question F1
            self.scorer3(logits, labels)
            __, _, ex_f1 = self.scorer3.get_metric(reset=True)
            f1s.append(ex_f1)

            # EM
            preds = logits.argmax(dim=-1)
            ex_em = (torch.eq(preds, labels).sum() == preds.nelement()).item()
            ems.append(ex_em)
        em = sum(ems) / len(ems)
        qst_f1 = sum(f1s) / len(f1s)

        if reset:
            self._score_tracker = collections.defaultdict(list)

        return {"ans_f1": ans_f1, "qst_f1": qst_f1, "em": em, "avg": (ans_f1 + em) / 2}


@register_task("record", rel_path="ReCoRD/")
class ReCoRDTask(Task):
    """Reading Comprehension with commonsense Reasoning Dataset
    See paper at https://sheng-z.github.io/ReCoRD-explorer """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, **kw)
        self.val_metric = "%s_avg" % self.name
        self.val_metric_decreases = False
        self._score_tracker = collections.defaultdict(list)
        self._answers = None
        self.max_seq_len = max_seq_len
        self.files_by_split = {
            "train": os.path.join(path, "train.jsonl"),
            "val": os.path.join(path, "val.jsonl"),
            "test": os.path.join(path, "test.jsonl"),
        }

    def load_data(self):
        # Data is exposed as iterable: no preloading
        pass

    def get_split_text(self, split: str):
        """ Get split text as iterable of records.

        Split should be one of "train", "val", or "test".
        """
        return self.load_data_for_path(self.files_by_split[split], split)

    def load_data_for_path(self, path, split):
        """ Load data """

        def tokenize_preserve_placeholder(sent, max_ent_length):
            """ Tokenize questions while preserving @placeholder token """
            sent_parts = sent.split("@placeholder")
            assert len(sent_parts) == 2
            placeholder_loc = len(
                tokenize_and_truncate(
                    self.tokenizer_name, sent_parts[0], self.max_seq_len - max_ent_length
                )
            )
            sent_tok = tokenize_and_truncate(
                self.tokenizer_name, sent, self.max_seq_len - max_ent_length
            )
            return sent_tok[:placeholder_loc] + ["@placeholder"] + sent_tok[placeholder_loc:]

        examples = []
        data = [json.loads(d) for d in open(path, encoding="utf-8")]
        for item in data:
            psg_id = item["idx"]
            psg = tokenize_and_truncate(
                self.tokenizer_name, item["passage"]["text"], self.max_seq_len
            )
            ent_idxs = item["passage"]["entities"]
            ents = [item["passage"]["text"][idx["start"] : idx["end"] + 1] for idx in ent_idxs]
            max_ent_length = max([idx["end"] - idx["start"] + 1 for idx in ent_idxs])
            qas = item["qas"]
            for qa in qas:
                qst = tokenize_preserve_placeholder(qa["query"], max_ent_length)
                qst_id = qa["idx"]
                if "answers" in qa:
                    anss = [a["text"] for a in qa["answers"]]
                else:
                    anss = []
                ex = {
                    "passage": psg,
                    "ents": ents,
                    "query": qst,
                    "answers": anss,
                    "psg_id": f"{split}-{psg_id}",
                    "qst_id": qst_id,
                }
                examples.append(ex)

        return examples

    def _load_answers(self) -> None:
        answers = {}
        for split, split_path in self.files_by_split.items():
            data = [json.loads(d) for d in open(split_path, encoding="utf-8")]
            for item in data:
                psg_id = f"{split}-{item['idx']}"
                for qa in item["qas"]:
                    qst_id = qa["idx"]
                    if "answers" in qa:
                        answers[(psg_id, qst_id)] = [a["text"] for a in qa["answers"]]
                    else:
                        answers[(psg_id, qst_id)] = ["No answer"]
        self._answers = answers

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for example in self.load_data_for_path(path, split):
                yield example["passage"]
                yield example["query"]

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def is_answer(x, ys):
            """ Given a list of answers, determine if x is an answer """
            return x in ys

        def insert_ent(ent, template):
            """ Replace ent into template (query with @placeholder) """
            assert "@placeholder" in template, "No placeholder detected!"
            split_idx = template.index("@placeholder")
            return template[:split_idx] + ent + template[split_idx + 1 :]

        def _make_instance(psg, qst, ans_str, label, psg_idx, qst_idx, ans_idx):
            """ pq_id: passage-question ID """
            d = {}
            d["psg_str"] = MetadataField(" ".join(psg))
            d["qst_str"] = MetadataField(" ".join(qst))
            d["ans_str"] = MetadataField(ans_str)
            d["psg_idx"] = MetadataField(par_idx)
            d["qst_idx"] = MetadataField(qst_idx)
            d["ans_idx"] = MetadataField(ans_idx)
            d["idx"] = MetadataField(ans_idx)  # required by evaluate()
            if model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                inp = model_preprocessing_interface.boundary_token_fn(psg, qst)
                d["psg_qst_ans"] = sentence_to_text_field(inp, indexers)
            else:
                d["psg"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(psg), indexers
                )
                d["qst"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(qst), indexers
                )
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)

            return Instance(d)

        for example in split:
            psg = example["passage"]
            qst_template = example["query"]

            ent_strs = example["ents"]
            ents = [
                tokenize_and_truncate(self._tokenizer_name, ent, self.max_seq_len)
                for ent in ent_strs
            ]

            anss = example["answers"]
            par_idx = example["psg_id"]
            qst_idx = example["qst_id"]
            for ent_idx, (ent, ent_str) in enumerate(zip(ents, ent_strs)):
                label = is_answer(ent_str, anss)
                qst = insert_ent(ent, qst_template)
                yield _make_instance(psg, qst, ent_str, label, par_idx, qst_idx, ent_idx)

    def count_examples(self):
        """ Compute here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            data = [json.loads(d) for d in open(split_path, encoding="utf-8")]
            example_counts[split] = sum([len(d["passage"]["entities"]) for d in data])
        self.example_counts = example_counts

    def update_metrics(self, logits, anss, idxs, tagmask=None):
        """ A batch of logits+answer strings and the questions they go with """
        logits = logits.detach().cpu()
        for idx, logit, ans in zip(idxs, logits, anss):
            self._score_tracker[idx].append((logit, ans))

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""

        # Load asnwers, used for computing metrics
        if self._answers is None:
            self._load_answers()

        ems, f1s = [], []
        for idx, logits_and_anss in self._score_tracker.items():
            golds = self._answers[idx]
            logits_and_anss.sort(key=lambda x: x[1])
            logits, anss = list(zip(*logits_and_anss))
            logits = torch.stack(logits)

            # take the most probable choice as the model prediction
            pred_idx = torch.softmax(logits, dim=-1)[:, -1].argmax().item()
            pred = anss[pred_idx]

            # F1
            f1 = metric_max_over_ground_truths(f1_score, pred, golds)
            f1s.append(f1)

            # EM
            em = metric_max_over_ground_truths(exact_match_score, pred, golds)
            ems.append(em)

        em = sum(ems) / len(ems)
        f1 = sum(f1s) / len(f1s)

        if reset:
            self._score_tracker = collections.defaultdict(list)

        return {"f1": f1, "em": em, "avg": (f1 + em) / 2}


@register_task("qasrl", rel_path="QASRL/")
class QASRLTask(SpanPredictionTask):
    def __init__(self, path, max_seq_len, name, **kw):
        """QA-SRL (Question-Answer Driven Semantic Role Labeling)
        See http://qasrl.org/
        Download, unzip, and rename the "qasrl-v2" folder to "QASRL"
        """
        super(QASRLTask, self).__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.f1_scorer = SpanF1Measure()
        self.val_metric = "%s_f1" % self.name
        self.val_metric_decreases = False

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        f1 = self.f1_scorer.get_metric(reset)
        return {"f1": f1}

    def update_metrics(self, logits, labels, tagmask=None):
        self.f1_scorer(
            pred_start=logits["span_start"],
            pred_end=logits["span_end"],
            gold_start=labels["span_start"],
            gold_end=labels["span_end"],
        )

    def load_data(self):
        self.train_data = self._load_file(os.path.join(self.path, "orig", "train.jsonl.gz"))

        self.val_data = self._load_file(os.path.join(self.path, "orig", "dev.jsonl.gz"))
        self._shuffle_data(self.val_data)

        self.test_data = self._load_file(os.path.join(self.path, "orig", "test.jsonl.gz"))
        self.sentences = []

        self.sentences = (
            self.train_data[0] + self.train_data[1] + self.val_data[0] + self.val_data[1]
        )

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        yield from self.sentences

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        def _make_instance(sentence_tokens, question_tokens, answer_span, idx):
            d = dict()

            # For human-readability
            d["raw_sentence"] = MetadataField(" ".join(sentence_tokens[1:-1]))
            d["raw_question"] = MetadataField(" ".join(question_tokens[1:-1]))

            if model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                inp = model_preprocessing_interface.boundary_token_fn(
                    sentence_tokens, question_tokens
                )
                d["inputs"] = sentence_to_text_field(inp, indexers)
            else:
                d["sentence"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(sentence_tokens), indexers
                )
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(question_tokens), indexers
                )

            d["span_start"] = NumericField(answer_span[0], label_namespace="span_start_labels")
            d["span_end"] = NumericField(answer_span[1], label_namespace="span_end_labels")
            d["idx"] = LabelField(idx, label_namespace="idxs", skip_indexing=True)
            return Instance(d)

        split = list(split)
        instances = map(_make_instance, *split)
        return instances

    def _load_file(self, path):
        example_list = []
        aligner_fn = get_aligner_fn(self.tokenizer_name)
        with gzip.open(path) as f:
            lines = f.read().splitlines()
            for line in lines:
                datum = self.preprocess_qasrl_datum(json.loads(line))
                sentence_tokens = datum["sentence_tokens"]
                # " ".join because retokenizer functions assume space-delimited input tokens
                aligner, processed_sentence_tokens = aligner_fn(" ".join(sentence_tokens))
                for entry in datum["entries"]:
                    for question, answer_list in entry["questions"].items():
                        for answer in answer_list:
                            for answer_span in answer:
                                projected_answer_span = aligner.project_span(*answer_span["span"])
                                # Adjust for [CLS] / <SOS> token
                                adjusted_answer_span = (
                                    projected_answer_span[0] + 1,
                                    projected_answer_span[1] + 1,
                                )
                                example_list.append(
                                    {
                                        "sentence_tokens": self._process_sentence(
                                            processed_sentence_tokens
                                        ),
                                        "question_tokens": self._process_sentence(question),
                                        "answer_span": adjusted_answer_span,
                                        "idx": len(example_list),
                                    }
                                )
        return [
            [example[k] for example in example_list]
            for k in ["sentence_tokens", "question_tokens", "answer_span", "idx"]
        ]

    @classmethod
    def _shuffle_data(cls, data, seed=1234):
        # Shuffle validation data to ensure diversity in periodic validation with val_data_limit
        indices = list(range(len(data[0])))
        random.Random(seed).shuffle(indices)
        return [[sub_data[i] for i in indices] for sub_data in data]

    def _process_sentence(self, sent):
        return tokenize_and_truncate(
            tokenizer_name=self.tokenizer_name, sent=sent, max_seq_len=self.max_seq_len
        )

    def get_split_text(self, split: str):
        return getattr(self, "%s_data" % split)

    @classmethod
    def preprocess_qasrl_datum(cls, datum):
        """ Extract relevant fields """
        return {
            "sentence_tokens": datum["sentenceTokens"],
            "entries": [
                {
                    "verb": verb_entry["verbInflectedForms"]["stem"],
                    "verb_idx": verb_idx,
                    "questions": {
                        question: [
                            [
                                {
                                    "tokens": datum["sentenceTokens"][span[0] : span[1] + 1],
                                    "span": span,
                                }
                                for span in answer_judgment["spans"]
                            ]
                            for answer_judgment in q_data["answerJudgments"]
                            if answer_judgment["isValid"]
                        ]
                        for question, q_data in verb_entry["questionLabels"].items()
                    },
                }
                for verb_idx, verb_entry in datum["verbEntries"].items()
            ],
        }


@register_task("commonsenseqa", rel_path="CommonsenseQA/")
@register_task("commonsenseqa-easy", rel_path="CommonsenseQA/", easy=True)
class CommonsenseQATask(MultipleChoiceTask):
    """ Task class for CommonsenseQA Task.  """

    def __init__(self, path, max_seq_len, name, easy=False, **kw):
        super().__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.easy = easy
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False
        self.n_choices = 5
        self.label2choice_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        self.choice_idx2label = ["A", "B", "C", "D", "E"]

    def load_data(self):
        """ Process the dataset located at path.  """

        def _load_split(data_file):
            questions, choices, targs, id_str = [], [], [], []
            data = [json.loads(l) for l in open(data_file, encoding="utf-8")]
            for example in data:
                question = tokenize_and_truncate(
                    self._tokenizer_name, "Q:" + example["question"]["stem"], self.max_seq_len
                )
                choices_dict = {
                    a_choice["label"]: tokenize_and_truncate(
                        self._tokenizer_name, "A:" + a_choice["text"], self.max_seq_len
                    )
                    for a_choice in example["question"]["choices"]
                }
                multiple_choices = [choices_dict[label] for label in self.choice_idx2label]
                targ = self.label2choice_idx[example["answerKey"]] if "answerKey" in example else 0
                example_id = example["id"]
                questions.append(question)
                choices.append(multiple_choices)
                targs.append(targ)
                id_str.append(example_id)
            return [questions, choices, targs, id_str]

        train_file = "train_rand_split_EASY.jsonl" if self.easy else "train_rand_split.jsonl"
        val_file = "dev_rand_split_EASY.jsonl" if self.easy else "dev_rand_split.jsonl"
        test_file = "test_rand_split_no_answers.jsonl"
        self.train_data_text = _load_split(os.path.join(self.path, train_file))
        self.val_data_text = _load_split(os.path.join(self.path, val_file))
        self.test_data_text = _load_split(os.path.join(self.path, test_file))
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + [choice for choices in self.train_data_text[1] for choice in choices]
            + [choice for choices in self.val_data_text[1] for choice in choices]
        )
        log.info("\tFinished loading CommonsenseQA data.")

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(question, choices, label, id_str):
            d = {}
            d["question_str"] = MetadataField(" ".join(question))
            if not model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(question), indexers
                )
            for choice_idx, choice in enumerate(choices):
                inp = (
                    model_preprocessing_interface.boundary_token_fn(question, choice)
                    if model_preprocessing_interface.model_flags["uses_pair_embedding"]
                    else model_preprocessing_interface.boundary_token_fn(choice)
                )
                d["choice%d" % choice_idx] = sentence_to_text_field(inp, indexers)
                d["choice%d_str" % choice_idx] = MetadataField(" ".join(choice))
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)
            d["id_str"] = MetadataField(id_str)
            return Instance(d)

        split = list(split)
        instances = map(_make_instance, *split)
        return instances

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        acc = self.scorer1.get_metric(reset)
        return {"accuracy": acc}
