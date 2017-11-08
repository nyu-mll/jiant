import pdb
import math
from random import shuffle
from abc import ABCMeta, abstractmethod, abstractproperty

import torch
import torch.nn as nn

from codebase.utils.seq_batch import SequenceBatch

# TODO(Alex)
# - sentiment task
# - RTE1-3 tasks
# - RTE8 task
# - Twitter humor + irony tasks

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

    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_data = None # TODO(Alex) what if tasks don't have test
        self.pred_layer = None # TODO(Alex): regularization + MLP option
        self.pair_input = None

    @abstractmethod
    def load_data(self, path): #TODO(Alex) what if tasks are across files
        '''
        Load data from path and create splits.
        '''
        raise NotImplementedError

    def _evaluate(self, model, data):
        '''
        Score model predictions against targets.

        Args:
            - model (TODO)
            - data (Dataset)

        Returns:
            - score (float): score averaged across examples
        '''
        score = 0.0
        for ins, targs in data:
            outs = model(ins, self.pred_layer, self.pair_input)
            _, preds = outs.max(1)
            pdb.set_trace()
            score += torch.sum(torch.eq(preds.long(), targs)).data[0]
        return score / len(data)

    def validate(self, model):
        '''
        Get validation scores for model
        '''
        return self._evaluate(model, self.val_data)

    def test(self, model):
        '''
        Get test scores for model
        '''
        return self._evaluate(model, self.test_data)

class Dataset():
    '''
    Data loader class for a single split of a dataset.
    A Task object will contain 2-3 Datasets.
    '''

    def __init__(self, inputs, targs, batch_size, tok2idx, pair_input):
        self.inputs = inputs
        self.targs = targs
        self._n_ins = len(targs)
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(self._n_ins / self.batch_size))
        self.batches = []
        for b_idx in range(self.n_batches):
            if pair_input:
                input0_batch = SequenceBatch.from_sequences(
                    inputs[0][b_idx*batch_size:(b_idx+1)*batch_size], tok2idx)
                input1_batch = SequenceBatch.from_sequences(
                    inputs[1][b_idx*batch_size:(b_idx+1)*batch_size], tok2idx)
                targs_batch = targs[b_idx*batch_size:(b_idx+1)*batch_size]
                self.batches.append(((input0_batch, input1_batch), targs_batch))
            else:
                input_batch = SequenceBatch.from_sequences(
                    inputs[b_idx*batch_size:(b_idx+1)*batch_size], tok2idx)
                targs_batch = targs[b_idx*batch_size:(b_idx+1)*batch_size]
                self.batches.append((input_batch, targs_batch))
        self._cur_batch = None

    def __len__(self):
        return self._n_ins

    def __iter__(self):
        self._cur_batch = 0
        return self

    def __next__(self):
        if self._cur_batch < self.n_batches:
            batch = self.batches[self._cur_batch]
            self._cur_batch += 1
            return batch
        else:
            raise StopIteration

class QuoraTask(Task):
    '''
    Task class for Quora question pairs.
    '''

    def __init__(self, path, input_dim, batch_size, cuda):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - n_classes (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(QuoraTask, self).__init__()
        self.pair_input = 1
        self.load_data(path)
        self.input_dim = input_dim
        self.n_classes = 2
        self.pred_layer = nn.Linear(input_dim, self.n_classes)
        if cuda:
            self.pred_layer = self.pred_layer.cuda()
        self.batch_size = batch_size

    def load_data(self, path):
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
        for raw_datum in raw_data:
            try:
                _, _, _, sent1, sent2, targ = \
                        raw_datum.split('\t')
            except Exception as e:
                print("Whoops")
                continue
            targ = int(targ)
            sents1.append(sent1), sents2.append(sent2), targs.append(targ)

        n_exs = len(sents1)
        split_pt1 = int(.8 * n_exs)
        split_pt2 = n_exs - int(.9 * n_exs) + split_pt1
        self.train_data_text = [sents1[:split_pt1], sents2[:split_pt1],
                                targs[:split_pt1]]
        self.val_data_text = [sents1[split_pt1:split_pt2],
                              sents2[split_pt1:split_pt2],
                              targs[split_pt1:split_pt2]]
        self.test_data_text = [sents1[split_pt2:], sents2[split_pt2:],
                               targs[split_pt2:]]
        # TODO(Alex): sort sentences by split by length



class SNLITask(Task):
    '''
    Task class for Stanford Natural Language Inference
    '''

    def __init__(self, path, input_dim, batch_size, cuda):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - n_classes (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(SNLITask, self).__init__()
        self.pair_input = 1
        self.load_data(path)
        self.input_dim = input_dim
        self.n_classes = 3
        self.pred_layer = nn.Linear(input_dim, self.n_classes)
        if cuda:
            self.pred_layer = self.pred_layer.cuda()
        self.batch_size = batch_size

    def load_data(self, path):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''
        for split, attr_name in zip(['train', 'dev', 'test'],
                                    ['train_data_text', 'val_data_text', 'test_data_text']):
            sents1, sents2, targs = [], [], []
            s1_fh = open(path + 's1.' + split)
            s2_fh = open(path + 's2.' + split)
            targ_fh = open(path + 'labels.' + split)
            for s1, s2, targ in zip(s1_fh, s2_fh, targ_fh):
                sents1.append(s1.strip())
                sents2.append(s2.strip())
                targ = targ.strip()
                if targ == 'neutral':
                    targs.append(0)
                if targ == 'entailment':
                    targs.append(1)
                if targ == 'contradiction':
                    targs.append(2)
            setattr(self, attr_name, [sents1, sents2, targs])

class MultiNLITask(Task):
    '''
    Task class for Multi-Genre Natural Language Inference
    '''

    def __init__(self, path, input_dim, batch_size, cuda):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - n_classes (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(SNLITask, self).__init__()
        self.pair_input = 1
        self.load_data(path)
        self.input_dim = input_dim
        self.n_classes = 3
        self.pred_layer = nn.Linear(input_dim, self.n_classes)
        if cuda:
            self.pred_layer = self.pred_layer.cuda()
        self.batch_size = batch_size

    def load_data(self, path):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''
        for split, attr_name in zip(['train', 'dev', 'test'],
                                    ['train_data_text', 'val_data_text', 'test_data_text']):
            sents1, sents2, targs = [], [], []
            s1_fh = open(path + 's1.' + split)
            s2_fh = open(path + 's2.' + split)
            targ_fh = open(path + 'labels.' + split)
            for s1, s2, targ in zip(s1_fh, s2_fh, targ_fh):
                sents1.append(s1.strip())
                sents2.append(s2.strip())
                targ = targ.strip()
                if targ == 'neutral':
                    targs.append(0)
                if targ == 'entailment':
                    targs.append(1)
                if targ == 'contradiction':
                    targs.append(2)
            setattr(self, attr_name, [sents1, sents2, targs])


class MSRPTask(Task):
    '''
    Task class for Quora question pairs.
    '''

    def __init__(self, path, input_dim, batch_size, cuda):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - n_classes (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(MSRPTask, self).__init__()
        self.pair_input = 1
        self.load_data(path)
        self.input_dim = input_dim
        self.n_classes = 2
        self.pred_layer = nn.Linear(input_dim, self.n_classes)
        if cuda:
            self.pred_layer = self.pred_layer.cuda()
        self.batch_size = batch_size

    def load_data(self, path):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''
        raise NotImplementedError


class STSTask(Task):
    '''
    Task class for Quora question pairs.
    '''

    def __init__(self, path, input_dim, batch_size, cuda):
        '''
        Args:
            - data (TODO)
            - input_dim (int)
            - n_classes (int)
            - classifier (str): type of classifier to use, log_reg or mlp
        '''
        super(STSTask, self).__init__()
        self.pair_input = 1
        self.load_data(path)
        self.input_dim = input_dim
        self.n_classes = 2
        self.pred_layer = nn.Linear(input_dim, self.n_classes)
        if cuda:
            self.pred_layer = self.pred_layer.cuda()
        self.batch_size = batch_size

    def load_data(self, path):
        '''
        Process the dataset located at path.

        TODO: preprocess and store data so don't have to wait?

        Args:
            - path (str): path to data
        '''
        raise NotImplementedError
