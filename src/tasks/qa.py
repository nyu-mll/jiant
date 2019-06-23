"""Task definitions for question answering tasks."""
import os
import re
import json
import collections
from typing import Iterable, Sequence, Type

import torch
from allennlp.training.metrics import Average, F1Measure
from allennlp.data.fields import LabelField, MetadataField
from allennlp.data import Instance

from ..utils.data_loaders import process_sentence

from .tasks import Task
from .tasks import sentence_to_text_field
from .registry import register_task

def _get_f1(x, y):
    """ """
    xs = x.split()
    ys = y.split()

    pcs = sum([1 for y in ys if y in xs]) / len(ys)
    rcl = sum([1 for x in xs if x in ys]) / len(xs)
    if pcs + rcl == 0:
        return 0
    else:
        return 2 * (pcs * rcl) / (pcs + rcl)

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
                # each example has a paragraph field -> (text, questions)
                # text is the paragraph, which requires some preprocessing
                # questions is a list of questions, has fields (question, sentences_used, answers)
                para = re.sub(
                    "<b>Sent .{1,2}: </b>", "", ex["paragraph"]["text"].replace("<br>", " ")
                )
                ex["paragraph"]["text"] = process_sentence(
                    self.tokenizer_name, para, self.max_seq_len
                )
                for question in ex["paragraph"]["questions"]:
                    question["question"] = process_sentence(
                        self.tokenizer_name, question["question"], self.max_seq_len
                    )
                    for answer in question["answers"]:
                        answer["text"] = process_sentence(
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
                yield example["paragraph"]["text"]
                for question in example["paragraph"]["questions"]:
                    yield question["question"]
                    for answer in question["answers"]:
                        yield answer["text"]

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        is_using_bert = "bert_wpm_pretokenized" in indexers

        def _make_instance(para, question, answer, label, par_idx, qst_idx, ans_idx):
            """ pq_id: paragraph-question ID """
            d = {}
            d["paragraph_str"] = MetadataField(" ".join(para))
            d["question_str"] = MetadataField(" ".join(question))
            d["answer_str"] = MetadataField(" ".join(answer))
            d["par_idx"] = MetadataField(par_idx)
            d["qst_idx"] = MetadataField(qst_idx)
            d["ans_idx"] = MetadataField(ans_idx)
            d["idx"] = MetadataField(ans_idx)  # required by evaluate()
            if is_using_bert:
                inp = para + question[1:-1] + answer[1:]
                d["para_quest_ans"] = sentence_to_text_field(inp, indexers)
            else:
                d["paragraph"] = sentence_to_text_field(para, indexers)
                d["question"] = sentence_to_text_field(question, indexers)
                d["answer"] = sentence_to_text_field(answer, indexers)
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)

            return Instance(d)

        for example in split:
            par_idx = example["idx"]
            para = example["paragraph"]["text"]
            for ex in example["paragraph"]["questions"]:
                qst_idx = ex["idx"]
                question = ex["question"]
                for answer in ex["answers"]:
                    ans_idx = answer["idx"]
                    ans = answer["text"]
                    label = int(answer["isAnswer"]) if "isAnswer" in answer else 0
                    yield _make_instance(para, question, ans, label, par_idx, qst_idx, ans_idx)

    def count_examples(self):
        """ Compute here b/c we"re streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            example_counts[split] = sum(
                len(q["answers"])
                for r in open(split_path, "r", encoding="utf-8")
                for q in json.loads(r)["paragraph"]["questions"]
            )

        self.example_counts = example_counts

    def update_metrics(self, logits, labels, idxs, tagmask=None):
        """ A batch of logits, labels, and the paragraph+questions they go with """
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
        self._answers = {} # Used for computing metrics, set when loading data
        self.max_seq_len = max_seq_len
        self.files_by_split = {
            "train": os.path.join(path, "train.json"),
            "val": os.path.join(path, "dev.json"),
            "test": os.path.join(path, "dev.json"),
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

        def split_then_tokenize(sent):
            """ Tokenize questions while preserving @placeholder token """
            sent_parts = sent.split("@placeholder")
            assert len(sent_parts) == 2
            sent_parts = [process_sentence(self.tokenizer_name, s, self.max_seq_len) for s in sent_parts]
            return sent_parts[0] + ["@placeholder"] + sent_parts[1]

        examples = []
        data = json.load(open(path, encoding="utf-8"))["data"]
        for item in data:
            psg_id = item["id"]
            psg = process_sentence(self.tokenizer_name, item["passage"]["text"],
                                   self.max_seq_len)
            ent_idxs = item["passage"]["entities"]
            ents = [psg[idx["start"]: idx["end"] + 1] for idx in ent_idxs]
            qas = item["qas"]
            for qa in qas:
                qst = split_then_tokenize(qa["query"])
                qst_id = qa["id"]
                anss = [a["text"] for a in qa["answers"]] # we don't use answer span info
                ex = {"passage": psg,
                      "ents": ents,
                      "query": qst,
                      "answers": anss,
                      "psg_id": psg_id,
                      "qst_id": qst_id
                     }
                self._answers[qst_id] = anss
                examples.append(ex)

        return examples

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        for split in self.files_by_split:
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for example in self.load_data_for_path(path):
                yield example["passage"]
                yield example["query"]

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """
        is_using_bert = "bert_wpm_pretokenized" in indexers

        def is_answer(x, ys):
            """ Given a list of answers, determine if x is an answer """
            return x in ys

        def insert_ent(ent, template):
            """ Replace ent into template """
            assert "@placeholder" in template, "No placeholder detected!"
            return [ent if t == "@placeholder" else t for t in template]

        def _make_instance(psg, qst, label, psg_idx, qst_idx, ans_idx):
            """ pq_id: paragraph-question ID """
            d = {}
            d["passage_str"] = MetadataField(" ".join(psg))
            d["question_str"] = MetadataField(" ".join(qst))
            d["answer_str"] = MetadataField(" ".join(answer))
            d["psg_idx"] = MetadataField(par_idx)
            d["qst_idx"] = MetadataField(qst_idx)
            d["ans_idx"] = MetadataField(ans_idx)
            d["idx"] = MetadataField(ans_idx) # required by evaluate()
            if is_using_bert:
                inp = para + question[1:]
                d["psg_qst"] = sentence_to_text_field(inp, indexers)
            else:
                d["passage"] = sentence_to_text_field(para, indexers)
                d["question"] = sentence_to_text_field(question, indexers)
            d["label"] = LabelField(label, label_namespace="labels", skip_indexing=True)

            return Instance(d)

        for example in split:
            psg = example["passage"]
            qst_template = example["query"]
            ents = example["ents"]
            anss = example["answers"]
            par_idx = example["psg_id"]
            qst_idx = example["psg_id"]
            for ent_idx, ent in enumerate(ents):
                label = is_answer(ent, anss)
                qst = insert_ent(ent, qst_template)
                yield _make_instance(psg, qst, label, par_idx, qst_idx, ent_idx)

    def count_examples(self):
        """ Compute here b/c we"re streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            data = json.load(open(split_path, "r", encoding="utf-8"))["data"]
            example_counts[split] = sum([len(d["passage"]["entities"]) for d in data])
        self.example_counts = example_counts

    def update_metrics(self, logits, anss, qst_idxs, tagmask=None):
        """ A batch of logits+answer strings and the questions they go with """
        logits = logits.detach().cpu()
        for idx, logit, ans in zip(qst_idxs, logits, anss):
            self._score_tracker[idx].append((logit, ans))

    def get_metrics(self, reset=False):
        """Get metrics specific to the task"""
        ems, f1s = [], []
        for qst_idx, logits_and_anss in self._score_tracker.items():
            golds = self._answers[qst_idx]
            logits, anss = list(zip(*logits_and_anss))
            logits = torch.stack(logits)
            pred = logits.argmax(dim=-1)
            pred = anss[pred.item()]

            # F1
            f1 = self._get_f1(pred, golds)
            f1s.append(f1)

            # EM
            em = max([int(pred == gold) for gold in golds])
            ems.append(em)

        em = sum(ems) / len(ems)
        f1 = sum(f1s) / len(f1s)

        return {"f1": f1, "em": em, "avg": (f1 + em) / 2}


