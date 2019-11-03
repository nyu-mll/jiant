""" 
Set of probing tasks that were added to Senteval Probing.
Paper: https://arxiv.org/abs/1805.01070
"""
import collections
import itertools
import json
import logging as log
import os

import numpy as np
import pandas as pd
import torch


# Fields for instance processing
from jiant.utils.data_loaders import tokenize_and_truncate
from jiant.tasks.registry import register_task  # global task registry
from jiant.tasks.tasks import SingleClassificationTask, process_single_pair_task_split


@register_task("se-probing-sentence-length", rel_path="sentence_length/")
class SEProbingSentenceLengthTask(SingleClassificationTask):
    """ Sentence length task   """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingSentenceLengthTask, self).__init__(name, n_classes=7, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return [str(x) for x in list(range(6))]

    def get_sentences(self):
        return self.sentences

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            rows = rows.sample(frac=1, axis=0).reset_index(drop=True)
            rows["s1"] = rows["2"].apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            return rows["s1"].tolist(), [], rows["1"].tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))

        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences


@register_task("se-probing-bigram-shift", rel_path="bigram_shift/")
class SEProbingBigramShiftTask(SingleClassificationTask):
    """  Bigram shift task   """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingBigramShiftTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return ["I", "O"]

    def get_sentences(self):
        return self.sentences

    def process_split(self, split, indexers, model_preprocessing_interface):
        return process_single_pair_task_split(
            split,
            indexers,
            model_preprocessing_interface,
            label_namespace=self._label_namespace,
            is_pair=False,
            skip_indexing=False,
        )

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            rows["s1"] = rows["2"].apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            return rows["s1"].tolist(), [], rows["1"].tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))

        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences


@register_task("se-probing-past-present", rel_path="past_present/")
class SEProbingPastPresentTask(SingleClassificationTask):
    """ Past Present Task  """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingPastPresentTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return ["PAST", "PRES"]

    def get_sentences(self):
        return self.sentences

    def process_split(self, split, indexers, model_preprocessing_interface):
        return process_single_pair_task_split(
            split,
            indexers,
            model_preprocessing_interface,
            label_namespace=self._label_namespace,
            is_pair=False,
            skip_indexing=False,
        )

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            rows["s1"] = rows["2"].apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            return rows["s1"].tolist(), [], rows["1"].tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))

        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences


@register_task("se-probing-odd-man-out", rel_path="odd_man_out/")
class SEProbingOddManOutTask(SingleClassificationTask):
    """ Odd man out task """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingOddManOutTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return ["C", "O"]

    def process_split(self, split, indexers, model_preprocessing_interface):
        return process_single_pair_task_split(
            split,
            indexers,
            model_preprocessing_interface,
            label_namespace=self._label_namespace,
            is_pair=False,
            skip_indexing=False,
        )

    def get_sentences(self):
        return self.sentences

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            rows["s1"] = rows["2"].apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            return rows["s1"].tolist(), [], rows["1"].tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))

        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences


@register_task("se-probing-coordination-inversion", rel_path="coordination_inversion/")
class SEProbingCoordinationInversionTask(SingleClassificationTask):
    """ Coordination Inversion task.  """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingCoordinationInversionTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return ["O", "I"]

    def get_sentences(self):
        return self.sentences

    def process_split(self, split, indexers, model_preprocessing_interface):
        return process_single_pair_task_split(
            split,
            indexers,
            model_preprocessing_interface,
            label_namespace=self._label_namespace,
            is_pair=False,
            skip_indexing=False,
        )

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            rows["s1"] = rows["2"].apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            return rows["s1"].tolist(), [], rows["1"].tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))

        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences


@register_task("se-probing-word-content", rel_path="word_content")
class SEProbingWordContentTask(SingleClassificationTask):
    """ Word Content Task  """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingWordContentTask, self).__init__(name, n_classes=1000, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return list(set(self.labels))

    def get_sentences(self):
        return self.sentences

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            rows["s1"] = rows["2"].apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            self.labels.append(rows["1"].tolist())
            return rows["s1"].tolist(), [], rows["1"].tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))

        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences


@register_task("se-probing-tree-depth", rel_path="tree_depth")
class SEProbingTreeDepthTask(SingleClassificationTask):
    """ Tree Depth Task """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingTreeDepthTask, self).__init__(name, n_classes=8, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return [str(x) for x in list(range(8))]

    def get_sentences(self):
        return self.sentences

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            labels = rows["1"].apply(lambda x: int(x.split("\t")[0]))
            labels = labels.apply(lambda x: x - 5)
            s1 = rows["1"].apply(lambda x: x.split("\t")[1])
            s1 = s1.apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            return s1.tolist(), [], labels.tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))
        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences


@register_task("se-probing-top-constituents", rel_path="top_constituents/")
class SEProbingTopConstituentsTask(SingleClassificationTask):
    """ Top Constituents task """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingTopConstituentsTask, self).__init__(name, n_classes=20, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return self.labels

    def process_split(self, split, indexers, model_preprocessing_interface):
        return process_single_pair_task_split(
            split,
            indexers,
            model_preprocessing_interface,
            label_namespace=self._label_namespace,
            is_pair=False,
            skip_indexing=False,
        )

    def get_sentences(self):
        return self.sentences

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            labels = rows["1"].apply(lambda x: str(x.split("\t")[0]))
            self.labels = list(set(labels.tolist()))
            s1 = rows["1"].apply(lambda x: x.split("\t")[1])
            s1 = s1.apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            return s1.tolist(), [], labels.tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))

        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences


@register_task("se-probing-subj-number", rel_path="subj_number")
class SEProbingSubjNumberTask(SingleClassificationTask):
    """ Subject number task """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingSubjNumberTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return ["NN", "NNS"]

    def process_split(self, split, indexers, model_preprocessing_interface):
        return process_single_pair_task_split(
            split,
            indexers,
            model_preprocessing_interface,
            label_namespace=self._label_namespace,
            is_pair=False,
            skip_indexing=False,
        )

    def get_sentences(self):
        return self.sentences

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            labels = rows["1"].apply(lambda x: str(x.split("\t")[0]))
            s1 = rows["1"].apply(lambda x: x.split("\t")[1])
            s1 = s1.apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            return s1.tolist(), [], labels.tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))

        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences


@register_task("se-probing-obj-number", rel_path="obj_number")
class SEProbingObjNumberTask(SingleClassificationTask):
    """ Object number task """

    def __init__(self, path, max_seq_len, name, **kw):
        super(SEProbingObjNumberTask, self).__init__(name, n_classes=2, **kw)
        self.path = path
        self.max_seq_len = max_seq_len
        self._label_namespace = self.name + "_tags"
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None

    def get_all_labels(self):
        return ["NN", "NNS"]

    def process_split(self, split, indexers, model_preprocessing_interface):
        return process_single_pair_task_split(
            split,
            indexers,
            model_preprocessing_interface,
            label_namespace=self._label_namespace,
            is_pair=False,
            skip_indexing=False,
        )

    def get_sentences(self):
        return self.sentences

    def load_data(self):
        """ Load data """

        def load_csv(data_file):
            rows = pd.read_csv(data_file, encoding="utf-8")
            labels = rows["1"].apply(lambda x: str(x.split("\t")[0]))
            s1 = rows["1"].apply(lambda x: x.split("\t")[1])
            s1 = s1.apply(
                lambda x: tokenize_and_truncate(self._tokenizer_name, x, self.max_seq_len)
            )
            return s1.tolist(), [], labels.tolist(), list(range(len(rows)))

        self.train_data_text = load_csv(os.path.join(self.path, "train.csv"))
        self.val_data_text = load_csv(os.path.join(self.path, "val.csv"))
        self.test_data_text = load_csv(os.path.join(self.path, "test.csv"))

        sentences = []
        for split in ["train", "val", "test"]:
            split_data = getattr(self, "%s_data_text" % split)
            sentences.extend(split_data[0])
        self.sentences = sentences
