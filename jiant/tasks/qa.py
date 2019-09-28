"""Task definitions for question answering tasks."""
import os
import re
import json
import string
import collections
from typing import Iterable, Sequence, Type

import torch
from allennlp.training.metrics import Average, F1Measure
from allennlp.data.fields import LabelField, MetadataField
from allennlp.data import Instance

from jiant.utils.data_loaders import tokenize_and_truncate

from jiant.tasks.tasks import Task
from jiant.tasks.tasks import sentence_to_text_field
from jiant.tasks.registry import register_task


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
        """ """
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
        """ """
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
