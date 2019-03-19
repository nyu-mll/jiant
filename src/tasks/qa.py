"""Task definitions for question answering tasks."""
import codecs
import collections
import logging as log
import json
import math
import os
import re
from typing import Iterable, Sequence, List, Dict, Any, Type

import allennlp.common.util as allennlp_util
from allennlp.training.metrics import Average, F1Measure
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import LabelField, MetadataField
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
        self.scorer2 = Average()
        self.val_metric = "%s_f1" % self.name
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
        def _make_instance(para, question, answer, label):
            d = {}
            inp = para + question[1:-1] + answer[1:]
            d["paragraph_question_answer"] = sentence_to_text_field(inp, indexers)
            d["paragraph_str"] = MetadataField(" ".join(para))
            d["question_str"] = MetadataField(" ".join(question))
            d["answer_str"] = MetadataField(" ".join(answer))
            d["labels"] = LabelField(label, skip_indexing=True)
            return Instance(d)

        for example in split:
            para = example["paragraph"]["text"]
            for ex in example["paragraph"]["questions"]:
                question = ex["question"]
                for answer in ex["answers"]:
                    #answers = [a["text"] for a in question["answers"]]
                    #labels = [int(a["isAnswer"]) for a in question["answers"]])
                    label = int(answer["isAnswer"])
                    yield _make_instance(para, question, answer["text"], label)

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

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        pcs, rcl, f1 = self.scorer1.get_metric(reset)
        question_acc = self.scorer2.get_metric(reset)
        return {'f1': f1} #, 'em': question_acc}

