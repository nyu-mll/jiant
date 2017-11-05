import pdb
import math
from random import shuffle
from abc import ABCMeta, abstractmethod, abstractproperty

import torch
import torch.nn as nn

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
    #__metaclass__ = ABCMeta

    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_data = None # TODO(Alex) what if tasks don't have test

    @abstractmethod
    def load_data(self, path): #TODO(Alex) what if tasks are across files
        '''
        Load data from path and create splits.
        '''
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, model, data):
        '''
        Method for scoring model predictions against targets
        '''
        raise NotImplementedError

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
    '''

    def __init__(self, inputs, targs, batch_size):
        self.inputs = inputs
        self.targs = targs
        self.n_ins = len(targs)
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(self.n_ins / self.batch_size))
        self.batch_idx = None

    def __iter__(self):
        self.batch_idx = 0
        return self

    def __next__(self):
        if self.batch_idx < self.n_batches:
            batch_inputs = [d[self.batch_idx*self.batch_size:
                              (self.batch_idx+1)*self.batch_size] \
                            for d in self.inputs]
            batch_targs = self.targs[self.batch_idx*self.batch_size:
                                     (self.batch_idx+1)*self.batch_size]
            self.batch_idx += 1
            return (batch_inputs, batch_targs)
        else:
            raise StopIteration

class QuoraTask(Task):
    '''
    Task class for Quora question pairs.
    '''

    def __init__(self, path, input_dim, batch_size):
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
        self.pred_layer = nn.Linear(input_dim * 4, self.n_classes).cuda()
        self.n_ins = len(self.train_data_text[0]) # TODO(Alex): filtered?
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(self.n_ins / self.batch_size))

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

    def _evaluate(self, model, data):
        '''
        Score model predictions against targets.

        Args:
            - outs (TODO)
            - targs (TODO)
        '''
        score = 0.0
        for ins, targs in data:
            outs = model(ins, self.pred_layer, self.pair_input)
            _, preds = outs.max(1)
            score += torch.sum(torch.eq(preds.long(), targs)).data[0]
        return score
