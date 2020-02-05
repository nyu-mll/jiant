"""Task definitions for question answering tasks."""
import os
import pandas as pd
import json
import collections
import gzip
import random
from typing import Iterable, Sequence, Type, Dict

import torch
import logging as log
from allennlp.training.metrics import Average, F1Measure, CategoricalAccuracy
from allennlp.data.fields import LabelField, MetadataField
from allennlp.data import Instance
from jiant.allennlp_mods.numeric_field import NumericField
from jiant.metrics.span_metrics import (
    metric_max_over_ground_truths,
    f1_score,
    exact_match_score,
    F1SpanMetric,
    ExactMatchSpanMetric,
)

from jiant.utils.data_loaders import tokenize_and_truncate
from jiant.utils.tokenizers import MosesTokenizer

from jiant.tasks.tasks import Task, SpanPredictionTask, MultipleChoiceTask
from jiant.tasks.tasks import sentence_to_text_field
from jiant.tasks.registry import register_task
from ..utils.retokenize import space_tokenize_with_spans, find_space_token_span, get_aligner_fn


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

    def update_metrics(self, out, batch):
        logits, labels, idxs = out["logits"], out["labels"], out["idxs"]
        tagmask = batch.get("taskmaster", None)
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

        examples = []
        data = [json.loads(d) for d in open(path, encoding="utf-8")]
        for item in data:
            psg_id = item["idx"]
            psg = tokenize_and_truncate(
                self.tokenizer_name, item["passage"]["text"], self.max_seq_len
            )
            ent_idxs = item["passage"]["entities"]
            ents = [item["passage"]["text"][idx["start"] : idx["end"] + 1] for idx in ent_idxs]
            qas = item["qas"]
            for qa in qas:
                qst = qa["query"]
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
            len(template.split("@placeholder")) == 2, "No placeholder detected!"
            return template.replace("@placeholder", ent)

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

            ents = example["ents"]

            anss = example["answers"]
            par_idx = example["psg_id"]
            qst_idx = example["qst_id"]
            for ent_idx, ent in enumerate(ents):
                label = is_answer(ent, anss)
                qst = tokenize_and_truncate(
                    self.tokenizer_name, insert_ent(ent, qst_template), self.max_seq_len
                )
                yield _make_instance(psg, qst, ent, label, par_idx, qst_idx, ent_idx)

    def count_examples(self):
        """ Compute here b/c we're streaming the sentences. """
        example_counts = {}
        for split, split_path in self.files_by_split.items():
            data = [json.loads(d) for d in open(split_path, encoding="utf-8")]
            example_counts[split] = sum([len(d["passage"]["entities"]) for d in data])
        self.example_counts = example_counts

    def update_metrics(self, out, batch):
        """ A batch of logits+answer strings and the questions they go with """
        logits, anss = out["logits"], out["anss"]
        idxs = [(p, q) for p, q in zip(batch["psg_idx"], batch["qst_idx"])]
        tagmask = batch.get("tagmask", None)
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

        self.f1_metric = F1SpanMetric()
        self.em_metric = ExactMatchSpanMetric()

        self.val_metric = "%s_avg" % self.name
        self.val_metric_decreases = False

    def count_examples(self, splits=["train", "val", "test"]):
        """ Count examples in the dataset. """
        pass

    def get_metrics(self, reset: bool = False) -> Dict:
        f1 = self.f1_metric.get_metric(reset)
        em = self.em_metric.get_metric(reset)
        collected_metrics = {"f1": f1, "em": em, "avg": (f1 + em) / 2}
        return collected_metrics

    def load_data(self):
        self.train_data = self._load_file(os.path.join(self.path, "orig", "train.jsonl.gz"))

        # Shuffle val_data to ensure diversity in periodic validation with val_data_limit
        self.val_data = self._load_file(
            os.path.join(self.path, "orig", "dev.jsonl.gz"), shuffle=True
        )

        self.test_data = self._load_file(os.path.join(self.path, "orig", "test.jsonl.gz"))

        self.sentences = (
            [example["passage"] for example in self.train_data]
            + [example["question"] for example in self.train_data]
            + [example["passage"] for example in self.val_data]
            + [example["question"] for example in self.val_data]
        )
        self.example_counts = {
            "train": len(self.train_data),
            "val": len(self.val_data),
            "test": len(self.test_data),
        }

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

    def _load_file(self, path, shuffle=False):
        example_list = []
        moses = MosesTokenizer()
        failed = 0
        with gzip.open(path) as f:
            lines = f.read().splitlines()

            for line in lines:
                datum = self.preprocess_qasrl_datum(json.loads(line))
                for entry in datum["entries"]:
                    for question, answer_list in entry["questions"].items():
                        for answer in answer_list:
                            for answer_span in answer:
                                answer_tok_span = (
                                    answer_span["span"][0],
                                    answer_span["span"][1] + 1,  # exclusive
                                )
                                try:
                                    remapped_result = remap_ptb_passage_and_answer_spans(
                                        ptb_tokens=datum["sentence_tokens"],
                                        answer_span=answer_tok_span,
                                        moses=moses,
                                        # We can move the aligned outside the loop, actually
                                        tokenizer_name=self.tokenizer_name,
                                    )
                                except ValueError:
                                    failed += 1
                                    continue
                                example_list.append(
                                    {
                                        "passage": self._process_sentence(
                                            remapped_result["detok_sent"]
                                        ),
                                        "question": self._process_sentence(question),
                                        "answer_span": remapped_result["answer_token_span"],
                                        "passage_str": remapped_result["detok_sent"],
                                        "answer_str": remapped_result["answer_str"],
                                        "space_processed_token_map": remapped_result[
                                            "space_processed_token_map"
                                        ],
                                    }
                                )

        if failed:
            log.info("FAILED ({}): {}".format(failed, path))

        if shuffle:
            random.Random(1234).shuffle(example_list)
        return example_list

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

    def get_metrics(self, reset: bool = False) -> Dict:
        f1 = self.f1_metric.get_metric(reset)
        em = self.em_metric.get_metric(reset)
        collected_metrics = {"f1": f1, "em": em, "avg": (f1 + em) / 2}
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
        for i, row in data_df.iterrows():
            # Answer indices are a space-limited list of numbers.
            # We simply take the min/max of the indices
            answer_idxs = list(map(int, row["answer"].split()))
            ans_tok_start, ans_tok_end = min(answer_idxs), max(answer_idxs) + 1  # Exclusive

            remapped_result = remap_ptb_passage_and_answer_spans(
                ptb_tokens=row["sent"].split(),
                answer_span=(ans_tok_start, ans_tok_end),
                moses=moses,
                tokenizer_name=self.tokenizer_name,
            )
            example_list.append(
                {
                    "passage": self._process_sentence(remapped_result["detok_sent"]),
                    "question": self._process_sentence(row["question"]),
                    "answer_span": remapped_result["answer_token_span"],
                    "passage_str": remapped_result["detok_sent"],
                    "answer_str": remapped_result["answer_str"],
                    "space_processed_token_map": remapped_result["space_processed_token_map"],
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
        self.example_counts = {
            "train": len(self.train_data),
            "val": len(self.val_data),
            "test": len(self.test_data),
        }

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


def remap_ptb_passage_and_answer_spans(ptb_tokens, answer_span, moses, tokenizer_name):
    # Start with PTB tokenized tokens
    # The answer_span is also in ptb_token space. We first want to detokenize, and convert everything to
    #   space-tokenization space.

    # Detokenize the passage. Everything we do will be based on the detokenized input,
    #   INCLUDING evaluation.
    detok_sent = moses.detokenize_ptb(ptb_tokens)

    # Answer indices are a space-limited list of numbers.
    # We simply take the min/max of the indices
    ans_tok_start, ans_tok_end = answer_span[0], answer_span[1]  # Exclusive
    # We convert the PTB-tokenized answer to char-indices.
    ans_char_start = len(moses.detokenize_ptb(ptb_tokens[:ans_tok_start]))
    while detok_sent[ans_char_start] == " ":
        ans_char_start += 1
    ans_char_end = len(moses.detokenize_ptb(ptb_tokens[:ans_tok_end]))
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
    aligner_fn = get_aligner_fn(tokenizer_name)
    token_aligner, actual_tokens = aligner_fn(detok_sent)

    # space_processed_token_map is a list of tuples
    #   (space_token, processed_token (e.g. BERT), space_token_index)
    # We will need this to map from token predictions to str spans
    space_processed_token_map = [
        (actual_tokens[actual_idx], space_token, space_idx)
        for space_idx, (space_token, _, _) in enumerate(space_tokens_with_spans)
        for actual_idx in token_aligner.project_tokens(space_idx)
    ]
    ans_actual_token_span = token_aligner.project_span(*ans_space_token_span)

    return {
        "detok_sent": detok_sent,
        "answer_token_span": ans_actual_token_span,
        "answer_str": answer_str,
        "space_processed_token_map": space_processed_token_map,
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


@register_task("cosmosqa", rel_path="cosmosqa/")
class CosmosQATask(MultipleChoiceTask):
    """ Task class for CosmosQA Task.
        adaptation of preprocessing from
        https://github.com/wilburOne/cosmosqa """

    def __init__(self, path, max_seq_len, name, **kw):
        super().__init__(name, **kw)
        self.path = path
        self.max_seq_len = max_seq_len

        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

        self.scorer1 = CategoricalAccuracy()
        self.scorers = [self.scorer1]
        self.val_metric = "%s_accuracy" % name
        self.val_metric_decreases = False
        self.n_choices = 4

    def load_data(self):
        """ Process the dataset located at path.  """
        self.train_data_text = self._load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = self._load_csv(os.path.join(self.path, "valid.csv"))
        self.test_data_text = self._load_csv(os.path.join(self.path, "test_no_label.csv"))
        self.sentences = (
            self.train_data_text[0]
            + self.val_data_text[0]
            + [choice for choices in self.train_data_text[1] for choice in choices]
            + [choice for choices in self.val_data_text[1] for choice in choices]
        )
        log.info("\tFinished loading CosmosQA data.")

    def _load_csv(self, input_file):
        import csv

        with open(input_file, "r") as csv_file:
            reader = csv.DictReader(csv_file)
            records = [record for record in reader]

        contexts, choices, targs, id_str = [], [], [], []
        for record in records:
            question = record["question"]

            ans_choices = [record["answer" + str(i)] for i in range(self.n_choices)]
            qa_tok_choices = [
                tokenize_and_truncate(
                    self._tokenizer_name, question + " " + ans_choices[i], self.max_seq_len
                )
                for i in range(len(ans_choices))
            ]
            max_ans_len = max([len(tok) for tok in qa_tok_choices])
            context = tokenize_and_truncate(
                self._tokenizer_name, record["context"], self.max_seq_len - max_ans_len
            )
            targ = int(record["label"]) if "label" in record else 0
            idx = record["id"]
            contexts.append(context)
            choices.append(qa_tok_choices)
            targs.append(targ)
            id_str.append(idx)
        return [contexts, choices, targs, id_str]

    def process_split(
        self, split, indexers, model_preprocessing_interface
    ) -> Iterable[Type[Instance]]:
        """ Process split text into a list of AllenNLP Instances. """

        def _make_instance(context, choices, label, id_str):
            d = {}
            d["context_str"] = MetadataField(" ".join(context))
            if not model_preprocessing_interface.model_flags["uses_pair_embedding"]:
                d["context"] = sentence_to_text_field(
                    model_preprocessing_interface.boundary_token_fn(context), indexers
                )
            for choice_idx, choice in enumerate(choices):
                inp = (
                    model_preprocessing_interface.boundary_token_fn(context, choice)
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
