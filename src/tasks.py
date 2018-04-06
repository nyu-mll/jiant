'''Define the tasks and code for loading them'''
import os
import pdb # pylint disable=unused-import
import xml.etree.ElementTree
import json
import glob
import codecs
import random
import logging as log
from abc import ABCMeta, abstractmethod
import nltk

from allennlp.training.metrics import CategoricalAccuracy, Average

def process_sentence(sent, max_seq_len):
    '''process a sentence using NLTK toolkit and adding SOS+EOS tokens'''
    #return ['<SOS>'] + nltk.word_tokenize(sent)[:max_seq_len] + ['<EOS>']
    return nltk.word_tokenize(sent)[:max_seq_len]

def load_tsv(data_file, max_seq_len, s1_idx=0, s2_idx=1, targ_idx=2, targ_map=None, targ_fn=None,
             skip_rows=0, delimiter='\t'):
    '''Load a tsv

    TODO: error handling; verifying parsed the TSV correctly (e.g. wrong # of cols)'''
    sent1s, sent2s, targs = [], [], []
    with codecs.open(data_file, 'r', 'utf-8') as data_fh:
        for _ in range(skip_rows):
            data_fh.readline()
        for row_idx, row in enumerate(data_fh):
            try:
                row = row.split(delimiter)
                sent1 = process_sentence(row[s1_idx], max_seq_len)
                if not row[targ_idx] or len(sent1) == 2:
                    continue
                if targ_map is not None:
                    targ = targ_map[row[targ_idx]]
                elif targ_fn is not None:
                    targ = targ_fn(row[targ_idx])
                else:
                    targ = int(row[targ_idx])
                if s2_idx is not None:
                    sent2 = process_sentence(row[s2_idx], max_seq_len)
                    if len(sent2) == 2:
                        continue
                    sent2s.append(sent2)
                sent1s.append(sent1)
                targs.append(targ)
            except Exception as e:
                print(e, row_idx)
                continue
    return sent1s, sent2s, targs

def split_data(data, ratio, shuffle=1):
    '''Split dataset according to ratio, larger split is first return'''
    n_exs = len(data[0])
    split_pt = int(n_exs * ratio)
    splits = [[], []]
    for col in data:
        splits[0].append(col[:split_pt])
        splits[1].append(col[split_pt:])
    return tuple(splits[0]), tuple(splits[1])

class Task():
    '''Abstract class for a task

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
        self.test_data = None
        self.pred_layer = None
        self.pair_input = 1
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

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        return {'accuracy': self.scorer.get_metric(reset)}

class QuoraTask(Task):
    '''
    Task class for Quora question pairs.
    '''

    def __init__(self, path, max_seq_len, name="quora"):
        ''' '''
        super(QuoraTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''
        tr_data = load_tsv(os.path.join(path, 'quora_duplicate_questions_clean.tsv'), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        tr_data, val_data = split_data(tr_data, .8)
        te_data = load_tsv(os.path.join(path, 'quora_test.tsv'), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading Quora data.")

class SNLITask(Task):
    ''' Task class for Stanford Natural Language Inference '''

    def __init__(self, path, max_seq_len, name="snli"):
        ''' Args: '''
        super(SNLITask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''

        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        for split, attr_name in zip(['train', 'dev', 'test'],
                                    ['train_data_text', 'val_data_text', 'test_data_text']):
            sents1, sents2, targs = [], [], []
            s1_fh = open(path + 's1.' + split)
            s2_fh = open(path + 's2.' + split)
            targ_fh = open(path + 'labels.' + split)
            for s1, s2, targ in zip(s1_fh, s2_fh, targ_fh):
                sents1.append(process_sentence(s1.strip(), max_seq_len))
                sents2.append(process_sentence(s2.strip(), max_seq_len))
                targs.append(targ_map[targ.strip()])
            sorted_data = (sents1, sents2, targs)
            setattr(self, attr_name, sorted_data)

        # Use adversarial NLI data instead of SNLI test because SNLI isn't in our benchmark
        #te_data = load_tsv(os.path.join(path, "adversarial_nli.tsv"), max_seq_len,
        #                   s1_idx=6, s2_idx=7, targ_idx=8, targ_map=targ_map, skip_rows=1)
        #self.test_data_text = te_data
        log.info("\tFinished loading SNLI data.")

class MultiNLITask(Task):
    ''' Task class for Multi-Genre Natural Language Inference '''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''
        super(MultiNLITask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at path.'''
        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(os.path.join(path, 'multinli_1.0_train.txt'), max_seq_len,
                           s1_idx=5, s2_idx=6, targ_idx=0, targ_map=targ_map, skip_rows=1)
        val_data = load_tsv(os.path.join(path, 'multinli_1.0_dev_matched.txt'), max_seq_len,
                            s1_idx=5, s2_idx=6, targ_idx=0, targ_map=targ_map, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'multinli_1.0_test.txt'), max_seq_len,
                           s1_idx=5, s2_idx=6, targ_idx=0, targ_map=targ_map, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading MNLI data.")

class MSRPTask(Task):
    ''' Task class for Microsoft Research Paraphase Task.  '''

    def __init__(self, path, max_seq_len, name="msrp"):
        ''' '''
        super(MSRPTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''

        tr_data = load_tsv(os.path.join(path, 'msr_paraphrase_train.txt'), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=0, skip_rows=1)
        tr_data, val_data = split_data(tr_data, ratio=.8)
        te_data = load_tsv(os.path.join(path, 'msr_paraphrase_test.txt'), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading MSRP data.")

class STSBenchmarkTask(Task):
    ''' Task class for Sentence Textual Similarity Benchmark.  '''
    def __init__(self, path, max_seq_len, name="sts_benchmark"):
        ''' '''
        super(STSBenchmarkTask, self).__init__(name, 1)
        self.categorical = 0
        self.val_metric = "%s_accuracy" % self.name
        self.val_metric_decreases = False
        self.scorer = Average()
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''
        tr_data = load_tsv(os.path.join(path, 'sts-train.csv'), max_seq_len,
                           s1_idx=5, s2_idx=6, targ_idx=4, targ_fn=lambda x: float(x) / 5)
        val_data = load_tsv(os.path.join(path, 'sts-dev.csv'), max_seq_len,
                            s1_idx=5, s2_idx=6, targ_idx=4, targ_fn=lambda x: float(x) / 5)
        te_data = load_tsv(os.path.join(path, 'sts-test.csv'), max_seq_len,
                           s1_idx=5, s2_idx=6, targ_idx=4, targ_fn=lambda x: float(x) / 5)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading STS Benchmark data.")

    def get_metrics(self, reset=False):
        # NB: I think I call it accuracy b/c something weird in training
        return {'accuracy': self.scorer.get_metric(reset)}

class SSTTask(Task):
    ''' Task class for Stanford Sentiment Treebank.  '''
    def __init__(self, path, max_seq_len, name="sst"):
        ''' '''
        super(SSTTask, self).__init__(name, 2)
        self.pair_input = 0
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''
        tr_data = load_tsv(os.path.join(path, 'sentiment-train'), max_seq_len,
                           s1_idx=0, s2_idx=None, targ_idx=1)
        val_data = load_tsv(os.path.join(path, 'sentiment-dev'), max_seq_len,
                            s1_idx=0, s2_idx=None, targ_idx=1)
        te_data = load_tsv(os.path.join(path, 'sentiment-test'), max_seq_len,
                           s1_idx=0, s2_idx=None, targ_idx=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SST data.")

class RTETask(Task):
    ''' Task class for Recognizing Textual Entailment 1, 2, 3, 5 '''

    def __init__(self, path, max_seq_len, name="rte"):
        ''' '''
        super(RTETask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' Process the datasets located at path. '''

        def load_files(paths):
            '''Load all files for a split'''
            targ_map = {"YES": 0, "ENTAILMENT": 0, "TRUE": 0,
                        "NO": 1, "CONTRADICTION": 1, "FALSE": 1, "UNKNOWN": 1, }

            sents1, sents2, targs = [], [], []
            for path in paths:
                root = xml.etree.ElementTree.parse(path).getroot()
                for child in root:
                    sents1.append(process_sentence(child[0].text, max_seq_len))
                    sents2.append(process_sentence(child[1].text, max_seq_len))
                    if "entailment" in child.attrib.keys():
                        label = child.attrib["entailment"]
                    elif "value" in child.attrib.keys():
                        label = child.attrib["value"]
                    targs.append(targ_map[label])
                    assert len(sents1) == len(sents2) == len(targs), pdb.set_trace()
            return sents1, sents2, targs

        devs = ["RTE2_dev_stanford_fix.xml", "RTE3_pairs_dev-set-final.xml",
                "rte1dev.xml", "RTE5_MainTask_DevSet.xml"]
        tests = ["RTE2_test.annotated.xml", "RTE3-TEST-GOLD.xml",
                 "rte1_annotated_test.xml", "RTE5_MainTask_TestSet_Gold.xml"]

        tr_data = load_files([os.path.join(path, dev) for dev in devs])
        tr_data, val_data = split_data(tr_data, .8)
        te_data = load_files([os.path.join(path, test) for test in tests])
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading RTE{1,2,3}.")


class SQuADTask(Task):
    '''Task class for adversarial SQuAD'''
    def __init__(self, path, max_seq_len, name="squad"):
        super(SQuADTask, self).__init__(name, 2)
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

        tr_data = load_split(os.path.join(path, "adv_squad_train.json"))
        val_data = load_split(os.path.join(path, "adv_squad_dev.json"))
        te_data = load_split(os.path.join(path, "adv_squad_test.json"))
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading SQuAD.")

class AcceptabilityTask(Task):
    '''Class for Warstdadt acceptability task'''
    def __init__(self, path, max_seq_len, name="acceptability"):
        ''' '''
        super(AcceptabilityTask, self).__init__(name, 2)
        self.pair_input = 0
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        tr_data = load_tsv(os.path.join(path, "acceptability_train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=None, targ_idx=1)
        val_data = load_tsv(os.path.join(path, "acceptability_valid.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=None, targ_idx=1)
        te_data = load_tsv(os.path.join(path, "acceptability_test.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=None, targ_idx=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading Acceptability.")

class WinogradNLITask(Task):
    '''Class for Winograd NLI task'''
    def __init__(self, path, max_seq_len, name="winograd"):
        ''' '''
        super(WinogradNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        tr_data = load_tsv(os.path.join(path, "wnli_train.tsv"), max_seq_len)
        val_data = load_tsv(os.path.join(path, "wnli_valid.tsv"), max_seq_len)
        te_data = load_tsv(os.path.join(path, "wnli_test.tsv"), max_seq_len)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
        log.info("\tFinished loading Winograd.")

class AdversarialTask(Task):
    '''Class for adversarial examples'''
    def __init__(self, path, max_seq_len, name="adversarial"):
        '''
        Args:
            path: path to data directory
        '''
        super(AdversarialTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Load the data'''
        targ_map = {'entailment': 1, 'neutral': 0, 'contradiction': 2}
        te_data = load_tsv(os.path.join(path, "adversarial_nli.tsv"), max_seq_len,
                           s1_idx=6, s2_idx=7, targ_idx=8, targ_map=targ_map, skip_rows=1)
        self.test_data_text = te_data
        self.train_data_text = te_data
        self.val_data_text = te_data
        log.info("\tFinished loading adversarial.")

#######################################
# Non-benchmark tasks
#######################################

class DPRTask(Task):
    '''Definite pronoun resolution'''
    def __init__(self, path, max_seq_len, name="dpr"):
        super(DPRTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, data_file, max_seq_len):
        '''Load data'''
        with open(data_file) as data_fh:
            raw_data = data_fh.read()
        raw_data = [datum.split('\n') for datum in raw_data.split('\n\n')]

        targ_map = {'entailed': 1, 'not-entailed': 0}
        tr_data = [[], [], []]
        val_data = [[], [], []]
        te_data = [[], [], []]
        for raw_datum in raw_data:
            sent1 = process_sentence(raw_datum[2].split(':')[1], max_seq_len)
            sent2 = process_sentence(raw_datum[3].split(':')[1], max_seq_len)
            targ = targ_map[raw_datum[4].split(':')[1].strip()]
            split = raw_datum[5].split(':')[1].strip()
            if split == 'train':
                tr_data[0].append(sent1)
                tr_data[1].append(sent2)
                tr_data[2].append(targ)
            elif split == 'dev':
                val_data[0].append(sent1)
                val_data[1].append(sent2)
                val_data[2].append(targ)
            elif split == 'test':
                te_data[0].append(sent1)
                te_data[1].append(sent2)
                te_data[2].append(targ)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data

class RTE5Task(Task):
    '''
    Task class for Recognizing Textual Entailment 5.
    '''

    def __init__(self, path, max_seq_len, name="rte5"):
        '''
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
        log.info("\tFinished loading RTE5.")

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
        log.info("\tFinished loading RTE8 task.")

class STS14Task(Task):
    '''
    Task class for Sentence Textual Similarity 14.
    Training data is STS12 and STS13 data, as provided in the dataset.
    '''
    def __init__(self, path, max_seq_len, name="sts14"):
        ''' '''
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
        log.info("\tFinished loading STS14 data.")

    def get_metrics(self, reset=False):
        return {'accuracy': self.scorer.get_metric(reset)}


class TwitterIronyTask(Task):
    ''' Task class for SemEval2018 Task 3: recognizing irony.  '''

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
        ''' Process the datasets located at path.  '''

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
        log.info("\tFinished loading Twitter irony task.")
