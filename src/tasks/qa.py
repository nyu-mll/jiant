"""Task definitions for question answering tasks."""
import collections
import json
import os
import re
from typing import Iterable, Sequence, Type

import torch
from allennlp.training.metrics import Average, F1Measure
from allennlp.data.fields import LabelField, MetadataField
from allennlp.data import Instance

from ..utils.data_loaders import process_sentence

from .tasks import Task
from .tasks import sentence_to_text_field
from .registry import register_task


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
            d["idx"] = MetadataField(ans_idx) # required by evaluate()
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
