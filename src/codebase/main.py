import os
import pdb
import sys
import argparse
import logging as log
from collections import Counter
import _pickle as pkl
import nltk
#import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

PATH_TO_PKG = '../'
sys.path.append(os.path.join(os.path.dirname(__file__), PATH_TO_PKG))
from codebase.utils.encoders import MultiLayerRNNEncoder
from codebase.models import MultiTaskModel
from codebase.tasks import Task, Dataset, QuoraTask

PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/processed_data/' + \
              'mtl-sentence-representations/'
NAME2TASK = {'quora': QuoraTask}
NAME2PATH = {'quora': PATH_PREFIX + 'Quora/quora_small.tsv'}
SPECIALS = ['<pad>', '<unk>', '<s>', '</s>']

def process_tasks(task_names, input_dim, max_vocab_size, max_seq_len,
                  batch_size, cuda):
    '''
    Process the tasks.

    Args:
        - task_names (list[str]): list of task names

    Returns:
        - tasks (list[Task]): list of tasks
    '''

    tasks = []
    for name in task_names:
        assert name in NAME2TASK, 'Task not found!'
        tasks.append(NAME2TASK[name](NAME2PATH[name], input_dim, batch_size))

    tok2idx = build_vocabulary(tasks, max_vocab_size)

    max_seq_len -= 2 # adding SOS and EOS tokens
    for task in tasks:
        task.train_data = process_split(task.train_data_text,
                                        task.pair_input, tok2idx,
                                        max_seq_len, batch_size, cuda)
        task.val_data = process_split(task.val_data_text, task.pair_input,
                                      tok2idx, max_seq_len, batch_size, cuda)
        task.test_data = process_split(task.test_data_text, task.pair_input,
                                       tok2idx, max_seq_len, batch_size, cuda)

    return tasks

def process_split(split, pair_input, tok2idx, max_seq_len, batch_size, cuda):
    '''
    Convert a dataset of sentences into padded sequences of indices.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)
        - max_seq_len (int)

    Returns:
    '''
    inputs1 = Variable(torch.LongTensor([process_sentence(
        sent, tok2idx, max_seq_len) for sent in split[0]]))
    targs = Variable(torch.LongTensor(split[-1]))
    if cuda:
        inputs1 = inputs1.cuda()
        targs = targs.cuda()
    if pair_input:
        inputs2 = Variable(torch.LongTensor([process_sentence(
            sent, tok2idx, max_seq_len) for sent in split[1]]))
        if cuda:
            inputs2 = inputs2.cuda()
        processed_split = Dataset([inputs1, inputs2], targs, batch_size)
    else:
        processed_split = Dataset([inputs1], targs, batch_size)
    return processed_split

def process_sentence(sentence, tok2idx=None, max_seq_len=None):
    '''
    Process a single sentence.

    Args:
        - sentence (str)
        - all_lower (int): 1 if uncase all words
        - tok2idx (dict): if given, return the sentence as indices

    Returns:
        - toks (list[str]): a list of processed tokens or indexes
    '''

    # use a tokenizer
    toks = nltk.word_tokenize(sentence)
    if tok2idx is not None:
        toks = [tok2idx[t] if t in tok2idx else tok2idx['<unk>'] \
                for t in toks]
        toks = [tok2idx['<s>']] + toks[:max_seq_len] + [tok2idx['</s>']] + \
               [tok2idx['<pad>']] * (max_seq_len - len(toks))
    return toks

def build_vocabulary(tasks, max_vocab_size):
    '''
    Build vocabulary across training data of all tasks.

    Args:
        - tasks (list[Task])

    Returns:
        - tok2idx (dict)
    '''

    # count frequencies
    tok2freq = Counter()
    for task in tasks:
        for sent in task.train_data_text[0]:
            toks = process_sentence(sent)
            for tok in toks:
                tok2freq[tok] += 1

        if task.pair_input:
            for sent in task.train_data_text[1]:
                toks = process_sentence(sent)
                for tok in toks:
                    tok2freq[tok] += 1

    # special characters
    tok2idx = {}
    for special in SPECIALS:
        tok2idx[special] = len(tok2idx)

    # pick top words
    if not max_vocab_size:
        max_vocab_size = len(tok2freq)
    for tok, _ in tok2freq.most_common()[:max_vocab_size]:
        tok2idx[tok] = len(tok2idx)

    return tok2idx

def main(arguments):
    '''
    Train or load a model. Evaluate on some tasks.
    '''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--cuda', help='0 if no CUDA, else gpu id',
                        type=int, default=1)

    parser.add_argument('--log_file', help='file to log to',
                        type=str, default=0)
    parser.add_argument('--data_path', help='path to directory containing '
                        ' {train,val,test}.pkl', type=str, default='')
    parser.add_argument('--word_embs_file', help='file containing word ' +
                        'embeddings', type=str, default='')

    parser.add_argument('--tasks', help='comma separated list of tasks',
                        type=str, default='')
    parser.add_argument('--max_vocab_size', help='vocabulary size',
                        type=int, default=10000)
    parser.add_argument('--max_seq_len', help='max sequence length',
                        type=int, default=40)

    parser.add_argument('--word_dim', help='dimension of word embeddings',
                        type=int, default=300)
    parser.add_argument('--hid_dim', help='hidden dimension size',
                        type=int, default=300)
    parser.add_argument('--n_layers', help='number of RNN layers',
                        type=int, default=1)

    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=10)
    parser.add_argument('--optimizer', help='optimizer to use',
                        type=str, default='sgd')
    parser.add_argument('--n_epochs', help='n epochs to train for',
                        type=int, default=10)
    parser.add_argument('--lr', help='starting learning rate',
                        type=float, default=1.0)

    args = parser.parse_args(arguments)

    log.basicConfig(format='%(asctime)s: %(message)s', level=log.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = log.FileHandler(args.log_file)
    log.getLogger().addHandler(file_handler)
    log.info(args)

    tasks = process_tasks(args.tasks.split(','), args.hid_dim,
                          args.max_vocab_size, args.max_seq_len,
                          args.batch_size, args.cuda)
    log.info('Finished loading tasks')

    token_embedder = nn.Embedding(args.max_vocab_size, args.word_dim,
                                  padding_idx=0)
    encoder = MultiLayerRNNEncoder(args.word_dim, args.hid_dim,
                                   args.n_layers, nn.LSTMCell)
    if args.cuda:
        token_embedder = token_embedder.cuda()
        encoder = encoder.cuda()
    model = MultiTaskModel(encoder, token_embedder, tasks)
    log.info('Finished building model')
    model.train_model(args.n_epochs, args.optimizer, args.lr)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
