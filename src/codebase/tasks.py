import os
import xml.etree.ElementTree
import glob
import pdb # pylint disable=unused-import
import math
import random
import logging as log
from collections import Counter
from random import shuffle
from abc import ABCMeta, abstractmethod, abstractproperty
import nltk
import _pickle as pkl

import torch
import torch.nn as nn
from allennlp.training.metrics import CategoricalAccuracy, Average

# TODO(Alex)
# - RTE1-3 tasks
# - Twitter humor
# - Twitter irony
# - DSTC

# TODO(Alex): put in another library
def process_sentence(sent, max_seq_len):
    return ['<s>'] + nltk.word_tokenize(sent)[:max_seq_len] + ['</s>']

class Task():
    '''
    Abstract class for a task

    Methods and attributes:
        - load_data: load dataset from a path and create splits
        - yield dataset for training
        - dataset size
        - validate and test

    Outside the task:
        - process: pad and indexify data given a mapping
        - optimizer
    '''
    __metaclass__ = ABCMeta

    def __init__(self, name, n_classes):
        self.name = name
        self.n_classes = n_classes
        self.train_data_text, self.val_data_text, self.test_data_text = \
            None, None, None
        self.train_data = None
        self.val_data = None
        self.test_data = None # TODO(Alex) what if tasks don't have test
        self.pred_layer = None
        self.pair_input = None
        self.categorical = 1 # most tasks are
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer = CategoricalAccuracy()

    @abstractmethod
    def load_data(self, path):
        '''
        Load data from path and create splits.
        '''
        raise NotImplementedError

    def count_words(self, path=None):
        '''
        Count the words in training split of data and return a Counter.
        If path is not None and the path exists, try to load from path.
        If path is not None and path doesn't exist, save to path.
        '''
        if path is not None and os.path.isfile(path):
            try:
                tok2freq = pkl.load(open(path, 'rb'))
                return tok2freq
            except Exception as e: # TODO(Alex) catch the specific error
                print("Unable to load counts from {0}".format(path))
        tok2freq = Counter()
        if self.train_data_text is None:
            raise ValueError('Train data split is not processed yet!')
        for sent in self.train_data_text[0]:
            for tok in sent:
                tok2freq[tok] += 1
        if self.pair_input:
            for sent in self.train_data_text[1]:
                for tok in sent:
                    tok2freq[tok] += 1
        if path is not None:
            pkl.dump(tok2freq, open(path, 'wb'))
        return tok2freq

    def get_metrics(self, reset=False):
        return {'accuracy': self.scorer.get_metric(reset)}



class QuoraTask(Task):
    '''
    Task class for Quora question pairs.
    '''

    def __init__(self, path, max_seq_len, name="quora"):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(QuoraTask, self).__init__(name, 2)
        self.name = name
        self.pair_input = 1
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''
        with open(path) as fh:
            raw_data = fh.readlines()

        _ = raw_data[0]
        raw_data = raw_data[1:]
        shuffle(raw_data)

        sents1, sents2, targs = [], [], []
        n_truncated = 0
        for raw_datum in raw_data:
            try:
                _, _, _, sent1, sent2, targ = \
                        raw_datum.split('\t')
            except Exception as e:
                continue
            sent1 = process_sentence(sent1, max_seq_len)
            sent2 = process_sentence(sent2, max_seq_len)
            # isn't correctly counting truncated sents
            n_truncated += int(len(sent1) > max_seq_len)
            n_truncated += int(len(sent2) > max_seq_len)
            if len(sent1) == 0 or len(sent2) == 0:
                continue # will break preprocessing
            targ = int(targ)
            sents1.append(sent1), sents2.append(sent2), targs.append(targ)

        n_exs = len(sents1)
        split_pt1 = int(.8 * n_exs)
        split_pt2 = n_exs - int(.9 * n_exs) + split_pt1

        sort_data = lambda s1, s2, t: sorted(zip(s1, s2, t),
                                             key=lambda x: (len(x[0]),
                                                            len(x[1])))
        tr_data = sort_data(sents1[:split_pt1], sents2[:split_pt1],
                            targs[:split_pt1])
        val_data = sort_data(sents1[split_pt1:split_pt2],
                             sents2[split_pt1:split_pt2],
                             targs[split_pt1:split_pt2])
        te_data = sort_data(sents1[split_pt2:], sents2[split_pt2:],
                            targs[split_pt2:])

        unpack = lambda x: [l for l in map(list, zip(*x))]
        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished loading Quora data. %d/%d sentences truncated",
                 n_truncated, len(sents1) + len(sents2))


class SNLITask(Task):
    '''
    Task class for Stanford Natural Language Inference
    '''

    def __init__(self, path, max_seq_len, name="snli"):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - n_classes (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(SNLITask, self).__init__(name, 3)
        self.name = name
        self.pair_input = 1
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''

        sort_data = lambda s1, s2, t: \
            sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))
        unpack = lambda x: [l for l in map(list, zip(*x))]

        for split, attr_name in zip(['train', 'dev', 'test'],
                                    ['train_data_text', 'val_data_text',
                                        'test_data_text']):
            sents1, sents2, targs = [], [], []
            s1_fh = open(path + 's1.' + split)
            s2_fh = open(path + 's2.' + split)
            targ_fh = open(path + 'labels.' + split)
            for s1, s2, targ in zip(s1_fh, s2_fh, targ_fh):
                sents1.append(process_sentence(s1.strip(), max_seq_len))
                sents2.append(process_sentence(s2.strip(), max_seq_len))
                targ = targ.strip()
                if targ == 'neutral':
                    targs.append(0)
                if targ == 'entailment':
                    targs.append(1)
                if targ == 'contradiction':
                    targs.append(2)
            sorted_data = sort_data(sents1, sents2, targs)
            setattr(self, attr_name, unpack(sorted_data))
        log.info("\tFinished loading SNLI data.")

class MultiNLITask(Task):
    '''
    Task class for Multi-Genre Natural Language Inference
    '''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - n_classes (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(MultiNLITask, self).__init__(name, 3)
        self.name = name
        self.pair_input = 1
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''

        def load_file(path):
            sents1, sents2, targs = [], [], []
            with open(path) as fh:
                fh.readline()
                for raw_datum in fh:
                    raw_datum = raw_datum.split('\t')
                    targ = raw_datum[0].strip()
                    if targ == 'neutral':
                        targs.append(0)
                    elif targ == 'entailment':
                        targs.append(1)
                    elif targ == 'contradiction':
                        targs.append(2)
                    else:
                        continue
                    sent1 = process_sentence(raw_datum[5], max_seq_len)
                    sent2 = process_sentence(raw_datum[6], max_seq_len)
                    sents1.append(sent1)
                    sents2.append(sent2)
            assert len(sents1) == len(sents2) == len(targs)
            return sents1, sents2, targs

        sort_data = lambda s1, s2, t: \
            sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))
        tr_data = sort_data(*load_file(
            os.path.join(path, 'multinli_1.0_train.txt')))

        # TODO(Alex): lazily creating a test set
        sents1, sents2, targs = load_file(
                os.path.join(path, 'multinli_1.0_dev_matched.txt'))
        n_exs = len(sents1)
        split_pt = int(.5 * n_exs)
        val_data = sort_data(sents1[split_pt:], sents2[split_pt:],
                             targs[split_pt:])
        te_data = sort_data(sents1[:split_pt], sents2[:split_pt],
                            targs[:split_pt])

        unpack = lambda x: [l for l in map(list, zip(*x))]
        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)

        log.info("\tFinished loading MNLI data.")


class MSRPTask(Task):
    '''
    Task class for Microsoft Research Paraphase Task.
    '''

    def __init__(self, path, max_seq_len, name="msrp"):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - n_classes (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(MSRPTask, self).__init__(name, 2)
        self.name = name
        self.pair_input = 1
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''

        def load_file(path):
            with open(path) as fh:
                raw_data = fh.readlines()

            sents1, sents2, targs = [], [], []
            for raw_datum in raw_data[1:]:
                try:
                    targ, _, _, sent1, sent2 = raw_datum.split('\t')
                except Exception as e:
                    print("Fucked up - broken example")
                    continue
                sent1 = process_sentence(sent1, max_seq_len)
                sent2 = process_sentence(sent2, max_seq_len)
                if len(sent1) == 2 or len(sent2) == 2: # for SOS/EOS tokens
                    continue # will break preprocessing
                targ = int(targ)
                sents1.append(sent1)
                sents2.append(sent2)
                targs.append(targ)
            return sents1, sents2, targs

        sort_data = lambda s1, s2, t: sorted(zip(s1, s2, t),
                                             key=lambda x: (len(x[0]),
                                                            len(x[1])))

        sents1, sents2, targs = load_file(
                os.path.join(path, 'msr_paraphrase_train.txt'))
        n_exs = len(sents1)
        # TODO(Alex): lazily creating a validation set; do x-validation
        split_pt = int(.9 * n_exs)

        tr_data = sort_data(sents1[:split_pt], sents2[:split_pt],
                            targs[:split_pt])
        val_data = sort_data(sents1[split_pt:], sents2[split_pt:],
                             targs[split_pt:])
        te_data = sort_data(*load_file(
            os.path.join(path, 'msr_paraphrase_test.txt')))

        unpack = lambda x: [l for l in map(list, zip(*x))]
        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)

        log.info("\tFinished loading MSRP data.")


class STS14Task(Task):
    '''
    Task class for Sentence Textual Similarity 14.
    Training data is STS12 and STS13 data, as provided in the dataset.
    '''
    def __init__(self, path, max_seq_len, name="sts14"):
        '''
        Args:
        '''
        super(STS14Task, self).__init__(name, 1)
        self.name = name
        self.pair_input = 1
        self.categorical = 0
        self.val_metric = "%s_pearson" % self.name
        self.val_metric_decreases = False
        self.scorer = Average()
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''

        def load_year(years, path):
            sents1, sents2, targs = [], [], []
            for year in years:
                topics = sts2topics[year]
                for topic in topics:
                    topic_sents1, topic_sents2, topic_targs = \
                            load_file(path + 'STS%d-en-test/' % year, topic)
                    sents1 += topic_sents1
                    sents2 += topic_sents2
                    targs += topic_targs
            assert len(sents1) == len(sents2) == len(targs)
            return sents1, sents2, targs

        def load_year_split(path, year):
            sents1, sents2, targs = [], [], []
            topics = sts2topics[year]
            for topic in topics:
                topic_sents1, topic_sents2, topic_targs = \
                        load_file(path, topic)
                sents1 += topic_sents1
                sents2 += topic_sents2
                targs += topic_targs
            assert len(sents1) == len(sents2) == len(targs)
            return sents1, sents2, targs

        def load_file(path, topic):
            sents1, sents2, targs = [], [], []
            with open(path + 'STS.input.%s.txt' % topic) as fh, \
                open(path + 'STS.gs.%s.txt' % topic) as gh:
                for raw_sents, raw_targ in zip(fh, gh):
                    raw_sents = raw_sents.split('\t')
                    sent1 = process_sentence(raw_sents[0], max_seq_len)
                    sent2 = process_sentence(raw_sents[1], max_seq_len)
                    if not sent1 or not sent2:
                        continue
                    sents1.append(sent1)
                    sents2.append(sent2)
                    targs.append(float(raw_targ) / 5) # rescale for cosine
            return sents1, sents2, targs

        sort_data = lambda s1, s2, t: \
            sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))
        unpack = lambda x: [l for l in map(list, zip(*x))]

        sts2topics = {
            12: ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', \
                    'surprise.SMTnews'],
            13: ['FNWN', 'headlines', 'OnWN'],
            14: ['deft-forum', 'deft-news', 'headlines', 'images', \
                    'OnWN', 'tweet-news']
            }

        sents1, sents2, targs = [], [], []
        train_dirs = [('STS2012-train', 12), ('STS2012-test', 12), ('STS2013-test', 13)]
        for train_dir, year in train_dirs:
            res = load_year_split(path + train_dir, year)
            sents1 += res[0]
            sents2 += res[1]
            targs += res[2]
        data = [(s1, s2, t) for s1, s2, t in zip(sents1, sents2, targs)]
        random.shuffle(data)
        sents1, sents2, targs = unpack(data)
        split_pt = int(.9 * len(sents1))
        tr_data = sort_data(sents1[split_pt:], sents2[split_pt:],
                            targs[split_pt:])
        val_data = sort_data(sents1[:split_pt], sents2[:split_pt],
                             targs[:split_pt])
        te_data = sort_data(*load_year_split(path, 14))

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)

        log.info("\tFinished loading STS data.")

    def get_metrics(self, reset=False):
        return {}

class STSBenchmarkTask(Task):
    '''
    Task class for Sentence Textual Similarity Benchmark.
    '''
    def __init__(self, path, max_seq_len, name="sts_benchmark"):
        '''
        Args:
        '''
        super(STS14Task, self).__init__(name, 1)
        self.name = name
        self.pair_input = 1
        self.categorical = 0
        self.val_metric = "%s_mse" % self.name
        self.val_metric_decreases = True
        self.scorer = Average()
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''

        def load_year(years, path):
            sents1, sents2, targs = [], [], []
            for year in years:
                topics = sts2topics[year]
                for topic in topics:
                    topic_sents1, topic_sents2, topic_targs = \
                            load_file(path + 'STS%d-en-test/' % year, topic)
                    sents1 += topic_sents1
                    sents2 += topic_sents2
                    targs += topic_targs
            assert len(sents1) == len(sents2) == len(targs)
            return sents1, sents2, targs

        def load_file(path, topic):
            sents1, sents2, targs = [], [], []
            with open(path + 'STS.input.%s.txt' % topic) as fh, \
                open(path + 'STS.gs.%s.txt' % topic) as gh:
                for raw_sents, raw_targ in zip(fh, gh):
                    raw_sents = raw_sents.split('\t')
                    sent1, sent2 = map(nltk.word_tokenize, raw_sents)
                    #sent1, sent2 = [process_sentence(raw_sent, max_seq_len) for
                    #                   \ raw_sent in raw_sents]
                    if not sent1 or not sent2:
                        continue
                    sents1.append(sent1)
                    sents2.append(sent2)
                    targs.append(float(raw_targ))
            return sents1, sents2, targs

        sort_data = lambda s1, s2, t: \
            sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))
        unpack = lambda x: [l for l in map(list, zip(*x))]

        sts2topics = {
            12: ['MSRpar', 'MSRvid', 'SMTeuroparl', 'surprise.OnWN', \
                    'surprise.SMTnews'],
            13: ['FNWN', 'headlines', 'OnWN'],
            14: ['deft-forum', 'deft-news', 'headlines', 'images', \
                    'OnWN', 'tweet-news']
            }

        sents1, sents2, targs = load_year([12, 13], path)
        data = [(s1, s2, t) for s1, s2, t in zip(sents1, sents2, targs)]
        random.shuffle(data)
        sents1, sents2, targs = unpack(data)
        split_pt = int(.9 * len(sents1))
        tr_data = sort_data(sents1[split_pt:], sents2[split_pt:],
                            targs[split_pt:])
        val_data = sort_data(sents1[:split_pt], sents2[:split_pt],
                             targs[:split_pt])
        te_data = sort_data(*load_year([14], path))

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)

        log.info("\tFinished loading STS data.")

    def get_metrics(self, reset=False):
        return {}


class SSTTask(Task):
    '''
    Task class for Stanford Sentiment Treebank.
    '''
    def __init__(self, path, max_seq_len, name="sst"):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - n_classes (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(SSTTask, self).__init__(name, 2)
        self.name = name
        self.pair_input = 0
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''

        n_truncated, n_sents = 0, 0
        def load_file(path):
            sents, targs = [], []
            #with open(path, 'r', encoding='utf-8') as fh:
            with open(path) as fh:
                for raw_datum in fh:
                    datum = raw_datum.strip().split('\t')
                    sents.append(process_sentence(datum[0], max_seq_len))
                    targs.append(int(datum[1]))
            return sents, targs

        sort_data = lambda s1, t: \
            sorted(zip(s1, t), key=lambda x: (len(x[0])))

        tr_data = sort_data(*load_file(
            os.path.join(path, 'sentiment-train')))
        val_data = sort_data(*load_file(
            os.path.join(path, 'sentiment-dev')))
        te_data = sort_data(*load_file(
            os.path.join(path, 'sentiment-test')))

        unpack = lambda x: [l for l in map(list, zip(*x))]
        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished loading SST data.")


class RTE8Task(Task):
    '''
    Task class for Recognizing Textual Entailment-8
    '''

    def __init__(self, path, max_seq_len, name="rte"):
        '''
        Args:
            path: path to RTE-8 data directory
            way_type: using 2way or 3way data
        '''
        accept = [2, 3]
        way_type = 3
        if way_type not in accept:
            assert "Needs to be either 2way or 3way"
        super(RTE8Task, self).__init__(name, way_type)
        self.name = name
        self.pair_input = 1
        self.load_data(path, way_type, max_seq_len)


    def load_data(self, path, way_type, max_seq_len):
        '''
        Process the datasets located at path.

        This merges data in the beetle and sciEntsBank subdirectories
        Also merges different types of test data (unseen answers, questions, and domains)
        '''

        test_splits = ['answers', 'questions']#, 'domains']

        def load_files(paths, type):
            data = {}
            for k in range(len(paths)):
                path = paths[k]
                root = xml.etree.ElementTree.parse(path).getroot()
                if type == 'beetle':
                    for i in range(len(root[1])):
                        pairID = root[1][i].attrib['id']
                        data[pairID] = []
                        sent1 = process_sentence(root[1][i].text, max_seq_len)
                        data[pairID].append(sent1) # sent1, reference sentence
                    for i in range(len(root[2])):
                        try:
                            matchID = root[2][i].attrib['answerMatch']
                            if matchID in data:
                                sent2 = process_sentence(root[2][i].text, max_seq_len)
                                data[matchID].append(sent2)
                                data[matchID].append(root[2][i].attrib['accuracy'])
                        except:
                            '''
                            pass when there isn't an ID indicating
                            the reference answer the student answer corresponds to
                            '''
                            pass
                else:
                    for i in range(len(root[2])):
                        pairID = root[2][i].attrib['id']
                        data[pairID] = []
                        sent1 = nltk.word_tokenize(root[1][0].text)
                        sent2 = nltk.word_tokenize(root[2][i].text)
                        data[pairID].append(sent1) # reference sentence
                        data[pairID].append(sent2) # student sentence
                        data[pairID].append(root[2][i].attrib['accuracy'])
            return data

        subdirs = ['beetle', 'sciEntsBank']
        way_type = '%dway' % way_type
        def get_paths(path, set, way_type, subdir):
            set_path = os.path.join(path, set, way_type, subdir)
            if set == 'training':
                return glob.glob(set_path+ '/*.xml')
            else:
                paths = [x[0] for x in os.walk(set_path)][1:]
                merged = []
                for p in paths:
                    if not sum([1 for split in test_splits if split in p]):
                        continue
                    merged += glob.glob(p + '/*.xml')
                return merged

        train_data = []
        test_data = []
        for sub in subdirs:
            train_data.append(load_files(get_paths(path, 'training', way_type, sub), sub))
            test_data.append(load_files(get_paths(path, 'test', way_type, sub), sub))

        targ_map = {'incorrect':0, 'correct':1, 'contradictory':2}
        def reformat(data):
            sents1, sents2, targs = [], [], []
            merged = data[0]
            merged.update(data[1])
            for k in merged.keys():
                try:
                    sents1.append(merged[k][0])
                    sents2.append(merged[k][1])
                    targs.append(targ_map[merged[k][2]])
                except:
                    pass
            return sents1, sents2, targs

        sort_data = lambda s1, s2, t: \
                sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))

        sents1, sents2, targs = reformat(train_data)
        n_exs = len(sents1)
        split_pt = int(.2 * n_exs)
        tr_data = sort_data(sents1[split_pt:], sents2[split_pt:], targs[split_pt:])
        val_data = sort_data(sents1[:split_pt], sents2[:split_pt], targs[:split_pt])

        sents1, sents2, targs = reformat(test_data)
        te_data = sort_data(sents1, sents2, targs)

        unpack = lambda x: [l for l in map(list, zip(*x))]

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished loading RTE8 task.")


class RTETask(Task):
    '''
    Task class for Recognizing Textual Entailment 1-3
    '''

    def __init__(self, path, max_seq_len, name="rte"):
        '''
        Args:
            path: path to RTE data directory
        '''
        super(RTETask, self).__init__(name, 3)
        self.name = name
        self.pair_input = 1
        self.load_data(path, max_seq_len)


    def load_data(self, path, max_seq_len):
        '''
        Process the datasets located at path.

        This merges data in the beetle and sciEntsBank subdirectories
        Also merges different types of test data (unseen answers, questions, and domains)
        '''
        def load_files(paths, type):
            data = {}
            for k in range(len(paths)):
                path = paths[k]
                root = xml.etree.ElementTree.parse(path).getroot()
                if type == 'beetle':
                    for i in range(len(root[1])):
                        pairID = root[1][i].attrib['id']
                        data[pairID] = []
                        sent1 = nltk.word_tokenize(root[1][i].text)
                        data[pairID].append(sent1) # sent1, reference sentence
                    for i in range(len(root[2])):
                        try:
                            matchID = root[2][i].attrib['answerMatch']
                            if matchID in data:
                                sent2 = nltk.word_tokenize(root[2][i].text)
                                data[matchID].append(sent2)
                                data[matchID].append(root[2][i].attrib['accuracy'])
                        except:
                            '''
                            pass when there isn't an ID indicating
                            the reference answer the student answer corresponds to
                            '''
                            pass
                else:
                    for i in range(len(root[2])):
                        pairID = root[2][i].attrib['id']
                        data[pairID] = []
                        sent1 = nltk.word_tokenize(root[1][0].text)
                        sent2 = nltk.word_tokenize(root[2][i].text)
                        data[pairID].append(sent1) # reference sentence
                        data[pairID].append(sent2) # student sentence
                        data[pairID].append(root[2][i].attrib['accuracy'])
            return data

        subdirs = ['beetle', 'sciEntsBank']
        def get_paths(path, set, way_type, subdir):
            set_path = os.path.join(path, set, way_type, subdir)
            if set == 'training':
                return glob.glob(set_path+ '/*.xml')
            else:
                paths = [x[0] for x in  os.walk(set_path)][1:]
                merged = []
                for path in paths:
                    merged += glob.glob(path + '/*.xml')
                return merged

        train_data = []
        test_data = []
        for sub in subdirs:
            train_data.append(load_files(get_paths(path, 'training', way_type, sub), sub))
            test_data.append(load_files(get_paths(path, 'test', way_type, sub), sub))

        def reformat(data):
            sents1, sents2, targs = [], [], []
            merged = data[0]
            merged.update(data[1])
            for k in merged.keys():
                try:
                    sents1.append(merged[k][0])
                    sents2.append(merged[k][1])
                    targs.append(merged[k][2])
                except:
                    pass
            return sents1, sents2, targs

        sort_data = lambda s1, s2, t: \
                sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))

        sents1, sents2, targs = reformat(train_data)
        n_exs = len(sents1)
        split_pt = int(.2 * n_exs)
        tr_data = sort_data(sents1[split_pt:], sents2[split_pt:], targs[split_pt:])
        val_data = sort_data(sents1[:split_pt], sents2[:split_pt], targs[:split_pt])

        sents1, sents2, targs = reformat(test_data)
        te_data = sort_data(sents1, sents2, targs)

        unpack = lambda x: [l for l in map(list, zip(*x))]

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished processing RTE 1-3.")
