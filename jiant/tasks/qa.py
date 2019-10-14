"""Task definitions for question answering tasks."""
import os
import pandas as pd
import re
import json
import string
import collections
import gzip
import random
from typing import Iterable, Sequence, Type, Dict

import torch
from allennlp.training.metrics import Average, F1Measure
from allennlp.data.fields import LabelField, MetadataField
from allennlp.data import Instance
from jiant.allennlp_mods.numeric_field import NumericField
from jiant.allennlp_mods.span_metrics import (
    metric_max_over_ground_truths,
    f1_score,
    exact_match_score,
    F1SpanMetric,
    ExactMatchSpanMetric,
)

from jiant.utils.data_loaders import tokenize_and_truncate
from jiant.utils.tokenizers import MosesTokenizer

from jiant.tasks.tasks import Task, SpanPredictionTask, TaggingTask
from jiant.tasks.tasks import sentence_to_text_field
from jiant.tasks.registry import register_task
from ..utils.retokenize import get_aligner_fn, space_tokenize_with_spans, find_space_token_span


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

        # Load answers, used for computing metrics
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

        # Shuffle val_data to ensure diversity in periodic validation with val_data_limit
        self.val_data = self._load_file(
            os.path.join(self.path, "orig", "dev.jsonl.gz"), shuffle=True
        )

        self.test_data = self._load_file(os.path.join(self.path, "orig", "test.jsonl.gz"))

        self.sentences = (
            self.train_data[0] + self.train_data[1] + self.val_data[0] + self.val_data[1]
        )

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        yield from self.sentences

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        is_using_pytorch_transformers = "pytorch_transformers_wpm_pretokenized" in indexers

        def _make_instance(sentence_tokens, question_tokens, answer_span, idx):
            d = dict()

            # For human-readability
            d["raw_sentence"] = MetadataField(" ".join(sentence_tokens[1:-1]))
            d["raw_question"] = MetadataField(" ".join(question_tokens[1:-1]))

            if model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                inp, start_offset, _ = model_preprocessing_interface.boundary_token_fn(
                    sentence_tokens, question_tokens, get_offset=True
                )
                d["inputs"] = sentence_to_text_field(inp, indexers)
            else:
                d["sentence"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(sentence_tokens), indexers
                )
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(question_tokens), indexers
                )

            d["span_start"] = NumericField(
                answer_span[0] + start_offset, label_namespace="span_start_labels"
            )
            d["span_end"] = NumericField(
                answer_span[1] + start_offset, label_namespace="span_end_labels"
            )
            d["start_offset"] = MetadataField(start_offset)
            d["idx"] = LabelField(idx, label_namespace="idxs", skip_indexing=True)
            return Instance(d)

        split = list(split)
        instances = map(_make_instance, *split)
        return instances

    def _load_file(self, path, shuffle=False):
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
                                example_list.append(
                                    {
                                        "sentence_tokens": self._process_sentence(
                                            processed_sentence_tokens
                                        ),
                                        "question_tokens": self._process_sentence(question),
                                        "answer_span": projected_answer_span,
                                        "idx": len(example_list),
                                    }
                                )
        if shuffle:
            random.Random(1234).shuffle(example_list)
        return [
            [example[k] for example in example_list]
            for k in ["sentence_tokens", "question_tokens", "answer_span", "idx"]
        ]

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


@register_task("qamr", rel_path="QAMR/")
class QAMRTask(SpanPredictionTask):
    """ Question-Answer Meaning Representation (QAMR)
        https://github.com/uwnlp/qamr
    """

    def __init__(self, path, max_seq_len, name="qamr", **kw):
        """ There are 1363 supertags in CCGBank without introduced token. """
        self.path = path
        super(QAMRTask, self).__init__(name, **kw)
        self.max_seq_len = max_seq_len

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.f1_metric = F1SpanMetric()
        self.em_metric = ExactMatchSpanMetric()

        self.val_metric = "%s_avg" % self.name
        self.val_metric_decreases = False

    def update_metrics(self, pred_str_list, gold_str_list, tagmask=None):
        """ A batch of logits+answer strings and the questions they go with """
        self.f1_metric(pred_str_list=pred_str_list, gold_str_list=gold_str_list)
        self.em_metric(pred_str_list=pred_str_list, gold_str_list=gold_str_list)

    def get_metrics(self, reset: bool = False) -> Dict:
        f1 = self.f1_metric.get_metric(reset)
        em = self.em_metric.get_metric(reset)
        collected_metrics = {"f1": f1, "em": em, "avg": f1 + em}
        return collected_metrics

    def get_sentences(self) -> Iterable[Sequence[str]]:
        """ Yield sentences, used to compute vocabulary. """
        yield from self.sentences

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        def _make_instance(example):
            d = dict()

            # For human-readability
            d["raw_passage"] = MetadataField(" ".join(example["passage"]))
            d["raw_question"] = MetadataField(" ".join(example["question"]))

            if model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                inp, start_offset, _ = model_preprocessing_interface.boundary_token_fn(
                    example["passage"], example["question"], get_offset=True
                )
                d["inputs"] = sentence_to_text_field(inp, indexers)
            else:
                d["passage"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(example["passage"]), indexers
                )
                d["question"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(example["question"]), indexers
                )
                start_offset = 0
            d["span_start"] = NumericField(
                example["answer_span"][0] + start_offset, label_namespace="span_start_labels"
            )
            d["span_end"] = NumericField(
                example["answer_span"][1] + start_offset, label_namespace="span_end_labels"
            )
            d["start_offset"] = MetadataField(start_offset)
            d["passage_str"] = MetadataField(example["passage_str"])
            d["answer_str"] = MetadataField(example["answer_str"])
            d["space_processed_token_map"] = MetadataField(example["space_processed_token_map"])
            return Instance(d)

        instances = map(_make_instance, split)
        return instances

    def get_split_text(self, split: str):
        return getattr(self, "%s_data" % split)

    @classmethod
    def load_tsv_dataset(cls, path, wiki_dict):
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=[
                "sent_id",
                "target_ids",
                "worker_id",
                "qa_index",
                "qa_word",
                "question",
                "answer",
                "response1",
                "response2",
            ],
        )
        df["sent"] = df["sent_id"].apply(wiki_dict.get)
        return df

    def process_dataset(self, data_df, shuffle=False):
        example_list = []
        moses = MosesTokenizer()
        aligner_fn = get_aligner_fn(self.tokenizer_name)
        for i, row in data_df.iterrows():
            # Start with PTB tokenized tokens
            tokens = row["sent"].split()

            # Detokenize the passage. Everything we do will be based on the detokenized input,
            #   INCLUDING evaluation.
            detok_sent = moses.detokenize_ptb(tokens)

            # Answer indices are a space-limited list of numbers.
            # We simply take the min/max of the indices
            answer_idxs = list(map(int, row["answer"].split()))
            ans_tok_start, ans_tok_end = min(answer_idxs), max(answer_idxs) + 1  # Exclusive
            # We convert the PTB-tokenized answer to char-indices.
            ans_char_start = len(moses.detokenize_ptb(tokens[:ans_tok_start]))
            ans_char_end = len(moses.detokenize_ptb(tokens[:ans_tok_end]))
            answer_str = detok_sent[ans_char_start:ans_char_end].strip()

            # We space-tokenize, with the accompanying char-indices.
            # We use the char-indices to map the answers to space-tokens.
            space_tokens_with_spans = space_tokenize_with_spans(detok_sent)
            ans_space_token_span = find_space_token_span(
                space_tokens_with_spans=space_tokens_with_spans,
                char_start=ans_char_start,
                char_end=ans_char_end,
            )
            # We project the space-tokenized answer to processed-tokens (e.g. BERT).
            # The latter is used for training/predicting.
            aligner, processed_sentence_tokens = aligner_fn(detok_sent)
            answer_token_span = aligner.project_span(*ans_space_token_span)

            # space_processed_token_map is a list of tuples
            #   (space_token, processed_token (e.g. BERT), space_token_index)
            # We will need this to map from token predictions to str spans
            space_processed_token_map = []
            for space_token_i, (space_token, char_start, char_end) in enumerate(
                space_tokens_with_spans
            ):
                processed_token_span = aligner.project_span(space_token_i, space_token_i + 1)
                for p_token_i in range(*processed_token_span):
                    space_processed_token_map.append(
                        (processed_sentence_tokens[p_token_i], space_token, space_token_i)
                    )

            example_list.append(
                {
                    "passage": self._process_sentence(detok_sent),
                    "question": self._process_sentence(row["question"]),
                    "answer_span": answer_token_span,
                    "passage_str": detok_sent,
                    "answer_str": answer_str,
                    "space_processed_token_map": space_processed_token_map,
                }
            )

        if shuffle:
            random.Random(12345).shuffle(example_list)

        return example_list

    def _process_sentence(self, sent):
        return tokenize_and_truncate(
            tokenizer_name=self.tokenizer_name, sent=sent, max_seq_len=self.max_seq_len
        )

    @classmethod
    def load_wiki_dict(cls, path):
        wiki_df = pd.read_csv(path, sep="\t", names=["sent_id", "text"])
        wiki_dict = {row["sent_id"]: row["text"] for _, row in wiki_df.iterrows()}
        return wiki_dict

    def load_data(self):
        wiki_dict = self.load_wiki_dict(os.path.join(self.path, "qamr/data/wiki-sentences.tsv"))
        self.train_data = self.process_dataset(
            self.load_tsv_dataset(
                path=os.path.join(self.path, "qamr/data/filtered/train.tsv"), wiki_dict=wiki_dict
            )
        )
        self.val_data = self.process_dataset(
            self.load_tsv_dataset(
                path=os.path.join(self.path, "qamr/data/filtered/dev.tsv"), wiki_dict=wiki_dict
            ),
            shuffle=True,
        )
        self.test_data = self.process_dataset(
            self.load_tsv_dataset(
                path=os.path.join(self.path, "qamr/data/filtered/test.tsv"), wiki_dict=wiki_dict
            )
        )

        self.sentences = (
            [example["passage"] for example in self.train_data]
            + [example["question"] for example in self.train_data]
            + [example["passage"] for example in self.val_data]
            + [example["question"] for example in self.val_data]
        )

    @staticmethod
    def collapse_contiguous_indices(ls):
        """
        [2, 3, 4, 5, 6, 7, 8] -> [(2, 9)]
        [1, 2, 4, 5] -> [(1, 3), (4, 6)]
        """
        if not ls:
            return []
        output = []
        start = None
        prev = None
        for n in ls:
            if start is None:
                start = n
                prev = n
            elif n == prev + 1:
                prev += 1
                continue
            else:
                output.append((start, prev + 1))  # exclusive
                start = n
                prev = n
        output.append((start, prev + 1))  # exclusive
        return output
