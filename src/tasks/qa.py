"""Task definitions for question answering tasks."""
import codecs
import collections
import logging as log
import json
import math
import os
import re
from typing import Iterable, Sequence, List, Dict, Any, Type

import torch
import allennlp.common.util as allennlp_util
from allennlp.training.metrics import Average, F1Measure
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import LabelField, MetadataField, ListField, SequenceLabelField
from allennlp.data import Instance, Token

from ..utils.data_loaders import process_sentence
from ..utils.utils import truncate

from .tasks import Task, sentence_to_text_field
from .tasks import sentence_to_text_field, atomic_tokenize
from .tasks import UNK_TOK_ALLENNLP, UNK_TOK_ATOMIC
from .registry import register_task

@register_task('multirc', rel_path='MultiRC/')
class MultiRCTask(Task):
    '''Multiple sentence reading comprehension Task'''

    def __init__(self, path, max_seq_len, name, **kw):
        ''' '''
        super().__init__(name, **kw)
        self.scorer1 = F1Measure(positive_label=1)
        self.scorer2 = Average() # to delete
        self.scorer3 = F1Measure(positive_label=1)
        self._score_tracker = collections.defaultdict(list)
        self.val_metric = "%s_avg_f1" % self.name
        self.val_metric_decreases = False
        self.max_seq_len = max_seq_len
        self.files_by_split = {split: os.path.join(path, "%s.json" % split) for
                               split in ["train", "val", "test"]}

    def tokenizer_is_supported(self, tokenizer_name):
        ''' For now, only support BERT '''
        return "bert" in tokenizer_name

    def get_split_text(self, split: str):
        ''' Get split text as iterable of records.

        Split should be one of 'train', 'val', or 'test'.
        '''
        return self.load_data(self.files_by_split[split])

    def load_data(self, path):
        ''' Load data '''
        data = json.load(open(path, "r"))["data"] # list of examples
        # each example has a paragraph field -> (text, questions)
        # text is the paragraph, which requires some preprocessing
        # questions is a list of questions, has fields (question, sentences_used, answers)
        for example in data:
            para = re.sub("<b>Sent .{1,2}: </b>", "", example["paragraph"]["text"].replace("<br>", " "))
            example["paragraph"]["text"] = process_sentence(self.tokenizer_name, para,
                                                            self.max_seq_len)
            for question in example["paragraph"]["questions"]:
                question["question"] = process_sentence(self.tokenizer_name, question["question"],
                                                        self.max_seq_len)
                for answer in question["answers"]:
                    answer["text"] = process_sentence(self.tokenizer_name, answer["text"],
                                                      self.max_seq_len)
        return data

    def get_sentences(self) -> Iterable[Sequence[str]]:
        ''' Yield sentences, used to compute vocabulary. '''
        for split in self.files_by_split:
            if split.startswith("test"):
                continue
            path = self.files_by_split[split]
            for example in self.load_data(path):
                yield example["paragraph"]["text"]
                for question in example["paragraph"]["questions"]:
                    yield question["question"]
                    for answer in question["answers"]:
                        yield answer["text"]

    def process_split(self, split, indexers) -> Iterable[Type[Instance]]:
        ''' Process split text into a list of AllenNLP Instances. '''
        def _make_instance(para, question, answer, label, pq_id):
            """ pq_id: paragraph-question ID """
            d = {}
            d["paragraph_str"] = MetadataField(" ".join(para))
            d["question_str"] = MetadataField(" ".join(question))
            d["par_quest_idx"] = MetadataField(pq_id)
            d["answer_str"] = MetadataField(" ".join(answer))
            inp = para + question[1:-1] + answer[1:]
            d["para_quest_ans"] = sentence_to_text_field(inp, indexers)
            d["label"] = LabelField(label, label_namespace="labels",
                                    skip_indexing=True)

            #answer_list = []
            #ans_str_list = []
            #for answer_n, (answer, label) in enumerate(zip(answers, labels)):
            #    inp = para + question[1:-1] + answer[1:]
            #    answer_list.append(sentence_to_text_field(inp, indexers))
            #    ans_str_list.append(MetadataField(" ".join(answer)))
            #d["answers"] = ListField(answer_list)
            #d["answer_strs"] = ListField(ans_str_list)
            #d["labels"] = SequenceLabelField(labels, d["answers"])
            return Instance(d)

        for par_n, example in enumerate(split):
            para = example["paragraph"]["text"]
            for quest_n, ex in enumerate(example["paragraph"]["questions"]):
                par_quest_n = "p%d_q%d" % (par_n, quest_n)
                question = ex["question"]
                #labels = [int(a["isAnswer"]) for a in ex["answers"]]
                #answers = [a["text"] for a in ex["answers"]]

                for answer in ex["answers"]:
                    label = int(answer["isAnswer"])
                    ans = answer["text"]
                    yield _make_instance(para, question, ans, label, par_quest_n)


    def count_examples(self):
        ''' Compute here b/c we're streaming the sentences. '''
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            #example_counts[split] = sum(len(ex["paragraph"]["questions"]) for ex in \
            #                            json.load(open(split_path, 'r'))["data"])
            example_counts[split] = sum(len(q["answers"]) for ex in \
                                        json.load(open(split_path, 'r'))["data"] for q in \
                                        ex["paragraph"]["questions"])

        self.example_counts = example_counts

    def update_metrics(self, logits, labels, idxs, tagmask=None):
        """ Expects logits and labels for all answers for a single question """
        self.scorer1(logits, labels)
        logits, labels = logits.detach().cpu(), labels.detach().cpu()
        # track progress on each question
        for ex, logit, label in zip(idxs, logits, labels):
            self._score_tracker[ex].append((logits, labels))

        #self.scorer3(logits, labels)
        #_, _, f1 = self.scorer3.get_metric(reset=True)
        #self.scorer2(f1)

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        _, _, ans_f1 = self.scorer1.get_metric(reset)

        f1s = []
        for logits_and_labels in self._score_tracker.values():
            logits, labels = list(zip(*logits_and_labels))
            logits = torch.stack(logits)
            labels = torch.stack(labels)
            self.scorer3(logits, labels)
            _, _, ex_f1 = self.scorer3.get_metric(reset=True)
            f1s.append(ex_f1)
        question_f1 = sum(f1s) / len(f1s)
        #question_f1 = self.scorer2.get_metric(reset)

        if reset:
            self._scorer_tracker = collections.defaultdict(list)

        return {'ans_f1': ans_f1, 'quest_f1': question_f1,
                "avg_f1": (ans_f1 + question_f1) / 2}

