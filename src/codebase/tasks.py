'''Define the tasks and code for loading them'''
import os
import pdb # pylint disable=unused-import
import xml.etree.ElementTree
import json
import glob
import random
import logging as log
from collections import Counter
from abc import ABCMeta, abstractmethod
import _pickle as pkl
import torch
import nltk

from allennlp.training.metrics import CategoricalAccuracy, Average

def process_sentence(sent, max_seq_len):
    '''process a sentence using NLTK toolkit and adding SOS+EOS tokens'''
    return ['<SOS>'] + nltk.word_tokenize(sent)[:max_seq_len] + ['<EOS>']

def shuffle(sent):
    raise NotImplementedError

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
        random.shuffle(raw_data)

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
            if not len(sent1) == 0 or not len(sent2) == 0:
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
        log.info("\tFinished processing Quora data.")


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
        log.info("\tFinished processing SNLI data.")

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

        log.info("\tFinished processing MNLI data.")


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
        log.info("\tFinished processing MSRP data.")


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
        #self.val_metric = "%s_accuracy" % self.name
        self.val_metric = "%s_accuracy" % self.name
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

        def load_year_split(path):
            sents1, sents2, targs = [], [], []
            input_files = glob.glob('%s/STS.input.*.txt' % path)
            targ_files = glob.glob('%s/STS.gs.*.txt' % path)
            input_files.sort()
            targ_files.sort()
            for inp, targ in zip(input_files, targ_files):
                topic_sents1, topic_sents2, topic_targs = \
                        load_file(path, inp, targ)
                sents1 += topic_sents1
                sents2 += topic_sents2
                targs += topic_targs
            assert len(sents1) == len(sents2) == len(targs)
            return sents1, sents2, targs

        def load_file(path, inp, targ):
            sents1, sents2, targs = [], [], []
            with open(inp) as fh, open(targ) as gh:
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
        train_dirs = ['STS2012-train', 'STS2012-test', 'STS2013-test']
        for train_dir in train_dirs:
            res = load_year_split(path + train_dir + '/')
            sents1 += res[0]
            sents2 += res[1]
            targs += res[2]
        data = [(s1, s2, t) for s1, s2, t in zip(sents1, sents2, targs)]
        random.shuffle(data)
        sents1, sents2, targs = unpack(data)
        split_pt = int(.8 * len(sents1))
        tr_data = sort_data(sents1[:split_pt], sents2[:split_pt],
                targs[:split_pt])
        val_data = sort_data(sents1[split_pt:], sents2[split_pt:],
                targs[split_pt:])
        te_data = sort_data(*load_year_split(path))

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished processing STS14 data.")

    def get_metrics(self, reset=False):
        return {'accuracy': self.scorer.get_metric(reset)}

class STSBenchmarkTask(Task):
    '''
    Task class for Sentence Textual Similarity Benchmark.
    '''
    def __init__(self, path, max_seq_len, name="sts_benchmark"):
        '''
        Args:
        '''
        super(STSBenchmarkTask, self).__init__(name, 1)
        self.name = name
        self.pair_input = 1
        self.categorical = 0
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases =  False
        self.scorer = Average()
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
                for row in fh:
                    row = row.split('\t')
                    score = float(row[-3]) / 5
                    sent1 = process_sentence(row[-2], max_seq_len)
                    sent2 = process_sentence(row[-1], max_seq_len)
                    if not sent1 or not sent2:
                        continue
                    sents1.append(sent1)
                    sents2.append(sent2)
                    targs.append(score)
            return sents1, sents2, targs

        sort_data = lambda s1, s2, t: \
            sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))
        unpack = lambda x: [l for l in map(list, zip(*x))]

        tr_data = sort_data(*load_file(path + 'sts-train.csv'))
        val_data = sort_data(*load_file(path + 'sts-dev.csv'))
        te_data = sort_data(*load_file(path + 'sts-test.csv'))

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished processing STS Benchmark data.")

    def get_metrics(self, reset=False):
        return {'accuracy': self.scorer.get_metric(reset)}


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
        log.info("\tFinished processing SST data.")


class RTE8Task(Task):
    '''
    Task class for Recognizing Textual Entailment-8
    '''

    def __init__(self, path, max_seq_len, name="rte8"):
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

        test_formats = ['questions'] #['answers', 'questions', 'domains']
        targ_map = {'incorrect':0, 'correct':1, 'contradictory':2}
        domains = ['beetle', 'sciEntsBank']
        way_type = '%dway' % way_type

        def get_paths(path, split, way_type, subdir):
            '''Get the xml files for a domain, split, and way split'''
            split_path = os.path.join(path, split, way_type, subdir)
            if split == 'training':
                paths = glob.glob(split_path + '/*.xml')
            else:
                filenames = [x[0] for x in os.walk(split_path)][1:]
                paths = []
                for filename in filenames:
                    if not sum([1 for test_format in test_formats if test_format in filename]):
                        continue
                    paths += glob.glob(filename + '/*.xml')
            return paths

        def load_files(paths, domain):
            sent1s, sent2s, targs = [], [], []
            missing = 0
            dbg = []
            for path in paths:
                root = xml.etree.ElementTree.parse(path).getroot()
                id2ref = {}
                question = root[0].text
                if domain == 'beetle':
                    for ref in root[1]: # reference answers
                        ref_id = ref.attrib['id']
                        ref_text = process_sentence(ref.text, max_seq_len)
                        id2ref[ref_id] = ref_text
                    for ans in root[2]: # student answers
                        try:
                            ref_id = ans.attrib['answerMatch']
                            if ref_id not in id2ref:
                                continue
                            ans_text = process_sentence(question + ans.text, max_seq_len)
                            ref_text = id2ref[ref_id]
                            targ = targ_map[ans.attrib['accuracy']]
                            sent1s.append(ans_text)
                            sent2s.append(ref_text)
                            targs.append(targ)
                        except KeyError:
                            '''
                            pass when there isn't an ID indicating
                            the reference answer the student answer corresponds to
                            '''
                            dbg.append((path, ans))
                            missing += 1
                else:
                    ref_text = process_sentence(root[1][0].text, max_seq_len)
                    for ans in root[2]:
                        ans_text = process_sentence(question + ans.text, max_seq_len)
                        targ = targ_map[ans.attrib['accuracy']]
                        sent1s.append(ans_text)
                        sent2s.append(ref_text)
                        targs.append(targ)
            #print("\t\tSkipped %d examples" % missing)
            return sent1s, sent2s, targs

        def do_everything(split, domains):
            '''For a split, for all domains, gather all the paths and process the files'''
            sent1s, sent2s, targs = [], [], []
            for domain in domains:
                paths = get_paths(path, split, way_type, domain)
                ret = load_files(paths, domain)
                sent1s += ret[0]
                sent2s += ret[1]
                targs += ret[2]
            return sent1s, sent2s, targs
        sort_data = lambda s1, s2, t: \
                sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))
        unpack = lambda x: [l for l in map(list, zip(*x))]

        sent1s, sent2s, targs = do_everything('training', domains)
        tmp = list(zip(sent1s, sent2s, targs))
        random.shuffle(tmp)
        sent1s, sent2s, targs = zip(*tmp)
        n_exs = len(sent1s)
        split_pt = int(.1 * n_exs)
        tr_data = sort_data(sent1s[split_pt:], sent2s[split_pt:], targs[split_pt:])
        val_data = sort_data(sent1s[:split_pt], sent2s[:split_pt], targs[:split_pt])
        sent1s, sent2s, targs = do_everything('test', domains)
        te_data = sort_data(sent1s, sent2s, targs)

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished processing RTE8 task.")



class RTETask(Task):
    '''
    Task class for Recognizing Textual Entailment 1, 2, and 3.
    '''

    def __init__(self, path, max_seq_len, name="rte"):
        '''
        Args:
            path: path to RTE data directory
        '''
        super(RTETask, self).__init__(name, 2)
        self.name = name
        self.pair_input = 1
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the datasets located at path.
        '''
        def load_files(paths):

            # Mapping the different label names to be consistent.
            LABEL_MAP = {
                "YES": 0,
                "ENTAILMENT": 0,
                "TRUE": 0,
                "NO": 1,
                "CONTRADICTION": 1,
                "FALSE": 1,
                "UNKNOWN": 1,
            }

            #data = {}
            sents1, sents2, targs = [], [], []
            for k in range(len(paths)):
                path = paths[k]
                root = xml.etree.ElementTree.parse(path).getroot()
                for i in range(len(root)):
                    sents1.append(process_sentence(root[i][0].text, max_seq_len))
                    sents2.append(process_sentence(root[i][1].text, max_seq_len))
                    if "entailment" in root[i].attrib.keys():
                        label = root[i].attrib["entailment"]
                    elif "value" in root[i].attrib.keys():
                        label = root[i].attrib["value"]
                    targs.append(LABEL_MAP[label])
                try:
                    assert len(sents1) == len(sents2) == len(targs)
                except AssertionError as e:
                    print(e)
                    pdb.set_trace()
            return sents1, sents2, targs

        devs = ["RTE2_dev_stanford_fix.xml", "RTE3_pairs_dev-set-final.xml",
                "rte1dev.xml", "RTE5_MainTask_DevSet.xml"]
        tests = ["RTE2_test.annotated.xml", "RTE3-TEST-GOLD.xml",
                 "rte1_annotated_test.xml", "RTE5_MainTask_TestSet_Gold.xml"]

        unpack = lambda x: [l for l in map(list, zip(*x))]
        sort_data = lambda s1, s2, t: \
                sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))

        # need to shuffle the data
        dev_sents1, dev_sents2, dev_targs = load_files([os.path.join(path, dev) for dev in devs])
        te_sents1, te_sents2, te_targs = load_files([os.path.join(path, test) for test in tests])

        n_exs = len(dev_sents1)
        split_pt = int(.2 * n_exs)
        tmp = list(zip(dev_sents1, dev_sents2, dev_targs))
        random.shuffle(tmp)
        dev_sents1, dev_sents2, dev_targs = zip(*tmp)
        tr_data = sort_data(dev_sents1[split_pt:], dev_sents2[split_pt:], dev_targs[split_pt:])
        val_data = sort_data(dev_sents1[:split_pt], dev_sents2[:split_pt], dev_targs[:split_pt])
        te_data = sort_data(te_sents1, te_sents2, te_targs)

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished processing RTE{1,2,3}.")

class RTE5Task(Task):
    '''
    Task class for Recognizing Textual Entailment 5.
    '''

    def __init__(self, path, max_seq_len, name="rte5"):
        '''
        Args:
            path: path to RTE data directory
        '''
        super(RTE5Task, self).__init__(name, 3)
        self.name = name
        self.pair_input = 1
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the datasets located at path.
        '''
        def load_files(paths):

            # Mapping the different label names to be consistent.
            LABEL_MAP = {
                "YES": 0,
                "ENTAILMENT": 0,
                "TRUE": 0,
                "NO": 1,
                "CONTRADICTION": 1,
                "FALSE": 1,
                "UNKNOWN": 2,
            }

            #data = {}
            sents1, sents2, targs = [], [], []
            for k in range(len(paths)):
                path = paths[k]
                root = xml.etree.ElementTree.parse(path).getroot()
                for i in range(len(root)):
                    sents1.append(process_sentence(root[i][0].text, max_seq_len))
                    sents2.append(process_sentence(root[i][1].text, max_seq_len))
                    if "entailment" in root[i].attrib.keys():
                        label = root[i].attrib["entailment"]
                    elif "value" in root[i].attrib.keys():
                        label = root[i].attrib["value"]
                    targs.append(LABEL_MAP[label])
                try:
                    assert len(sents1) == len(sents2) == len(targs)
                except AssertionError as e:
                    print(e)
                    pdb.set_trace()
            return sents1, sents2, targs

        devs = ["RTE5_MainTask_DevSet.xml"]
        tests = ["RTE5_MainTask_TestSet_Gold.xml"]

        unpack = lambda x: [l for l in map(list, zip(*x))]
        sort_data = lambda s1, s2, t: \
                sorted(zip(s1, s2, t), key=lambda x: (len(x[0]), len(x[1])))

        # need to shuffle the data
        dev_sents1, dev_sents2, dev_targs = load_files([os.path.join(path, dev) for dev in devs])
        te_sents1, te_sents2, te_targs = load_files([os.path.join(path, test) for test in tests])

        n_exs = len(dev_sents1)
        split_pt = int(.2 * n_exs)
        tmp = list(zip(dev_sents1, dev_sents2, dev_targs))
        random.shuffle(tmp)
        dev_sents1, dev_sents2, dev_targs = zip(*tmp)
        tr_data = sort_data(dev_sents1[split_pt:], dev_sents2[split_pt:], dev_targs[split_pt:])
        val_data = sort_data(dev_sents1[:split_pt], dev_sents2[:split_pt], dev_targs[:split_pt])
        te_data = sort_data(te_sents1, te_sents2, te_targs)

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished processing RTE5.")

class SQuADTask(Task):
    '''Task class for adversarial SQuAD'''
    def __init__(self, path, max_seq_len, name="squad"):
        super(SQuADTask, self).__init__(name, 2)
        self.name = name
        self.pair_input = 1
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Load the data'''

        def load_split(path):
            '''Load a single split'''
            quests, ctxs, targs = [], [], []
            data = json.load(open(path))
            for datum in data:
                quests.append(process_sentence(datum['question'], max_seq_len))
                ctxs.append(process_sentence(datum['sentence'], max_seq_len))
                assert datum['label'] in ['True', 'False'], pdb.set_trace()
                targs.append(int(datum['label'] == 'True'))
            return quests, ctxs, targs
            #return ctxs, targs

        tr_data = load_split(os.path.join(path, "adv_squad_train.json"))
        val_data = load_split(os.path.join(path, "adv_squad_dev.json"))
        te_data = load_split(os.path.join(path, "adv_squad_test.json"))
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished processing SQuAD.")


class TwitterIronyTask(Task):
    '''
    Task class for SemEval2018 Task 3: recognizing irony.
    '''

    def __init__(self, path, max_seq_len, name="twitter_irony"):
        '''
        Args:
            path: path to data directory
            way_type: using 2way or 3way data
        '''
        super(TwitterIronyTask, self).__init__(name, 2)
        self.name = name
        self.pair_input = 0
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''
        Process the datasets located at path.
        '''

        sents, targs = [], []
        with open(path) as fh:
            next(fh)
            for row in fh:
                row = row.split('\t')
                if len(row) > 3:
                    pdb.set_trace()
                targ = int(row[1])
                sent = process_sentence(row[2], max_seq_len)
                targs.append(targ)
                sents.append(sent)

        sort_data = lambda s1, t: sorted(zip(s1, t), key=lambda x: (len(x[0])))
        unpack = lambda x: [l for l in map(list, zip(*x))]

        n_exs = len(sents)
        tmp = list(zip(sents,targs))
        random.shuffle(tmp)
        sents, targs = zip(*tmp)
        split_pt1 = int(.8 * n_exs)
        split_pt2 = int(.9 * n_exs)
        tr_data = sort_data(sents[:split_pt1], targs[:split_pt1])
        val_data = sort_data(sents[split_pt1:split_pt2], targs[split_pt1:split_pt2])
        te_data = sort_data(sents[split_pt2:], targs[split_pt2:])

        self.train_data_text = unpack(tr_data)
        self.val_data_text = unpack(val_data)
        self.test_data_text = unpack(te_data)
        log.info("\tFinished processing Twitter irony task.")
