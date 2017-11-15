import os
import pdb
import sys
import time
import argparse
import logging as log
from collections import Counter
import _pickle as pkl
import nltk
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

PATH_TO_PKG = '../'
sys.path.append(os.path.join(os.path.dirname(__file__), PATH_TO_PKG))
from codebase.utils.encoders import MultiLayerRNNEncoder
from codebase.models import MultiTaskModel
from codebase.tasks import Task, Dataset, QuoraTask, SNLITask
from codebase.utils.seq_batch import SequenceBatch
from codebase.utils.token_embedder import TokenEmbedder
from codebase.utils.utils import GPUVariable

PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/processed_data/' + \
              'mtl-sentence-representations/'
NAME2TASK = {'quora': QuoraTask, 'snli': SNLITask, 'small':QuoraTask}
NAME2DATA = {'quora': PATH_PREFIX + 'Quora/quora_duplicate_questions.tsv',
             'snli': PATH_PREFIX + 'SNLI/',
             'small': PATH_PREFIX + 'Quora/quora_small.tsv'}

# lazy way to map tasks to vocab pickles
NAME2VOCAB = {QuoraTask: PATH_PREFIX + 'Quora/quora_vocab.pkl',
              SNLITask: PATH_PREFIX + 'SNLI/snli_vocab.pkl'}
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
        tasks.append(NAME2TASK[name](NAME2DATA[name], input_dim, max_seq_len,
                                     cuda))

    tok2idx = build_vocabulary(tasks, max_vocab_size)

    max_seq_len -= 2 # adding SOS and EOS tokens
    for task in tasks:
        task.train_data = process_split(task.train_data_text,
                                        task.pair_input, tok2idx,
                                        batch_size, cuda)
        task.val_data = process_split(task.val_data_text, task.pair_input,
                                      tok2idx, batch_size, cuda)
        task.test_data = process_split(task.test_data_text, task.pair_input,
                                       tok2idx, batch_size, cuda)

    return tasks, tok2idx

def process_split(split, pair_input, tok2idx, batch_size, cuda):
    '''
    Convert a dataset of sentences into padded sequences of indices.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)
        - max_seq_len (int)

    Returns:
    '''
    #inputs1 = Variable(torch.LongTensor([process_sentence(
    #    sent, tok2idx, max_seq_len) for sent in split[0]]))
    targs = GPUVariable(torch.LongTensor(split[-1]))
    if pair_input:
        processed_split = Dataset([split[0], split[1]], targs, batch_size,
                                  tok2idx, pair_input)

    else:
        processed_split = Dataset([split[0]], targs, batch_size, tok2idx,
                                  pair_input)
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
    '''
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
    '''
    tok2freq = Counter()
    counters = [task.count_words(NAME2VOCAB[task.__class__]) for task in tasks]
    for counter in counters:
        tok2freq.update(counter)

    # special characters
    tok2idx = {}
    for special in SPECIALS:
        tok2idx[special] = len(tok2idx)

    # pick top words
    if not max_vocab_size:
        max_vocab_size = len(tok2freq)
    else:
        max_vocab_size = max_vocab_size - len(SPECIALS)
    for tok, _ in tok2freq.most_common()[:max_vocab_size]:
        tok2idx[tok] = len(tok2idx)

    return tok2idx

def load_embeddings(path, tok2idx, word_dim):
    '''
    Load embeddings from GloVe vectors.

    TODO
        - make this a standard library function.

    Args:
        - path (str): path to word embedding file
        - tok2idx (dict): dictionary mapping words to indices
        - word_dim (int): word vector dimensionality

    Returns
        - embeddings (np.FloatArray): embeddings
    '''
    embeddings = np.random.rand(len(tok2idx), word_dim).astype(float)
    n_embs = 0
    with open(path) as fh:
        for row in fh:
            row = row.split()
            word = row[0]
            if word in tok2idx:
                assert len(row[1:]) == word_dim
                embeddings[tok2idx[word]] = np.array([float(v) for \
                                                     v in row[1:]])
                n_embs += 1

    embeddings[tok2idx['<pad>']] = 0.
    log.info("\tLoaded pretrained embeddings for {0} words".format(n_embs))
    return embeddings

def main(arguments):
    '''
    Train or load a model. Evaluate on some tasks.
    '''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--cuda', help='0 if no CUDA, else gpu id',
                        type=int, default=0)

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
                        type=int, default=4096)
    parser.add_argument('--n_layers', help='number of RNN layers',
                        type=int, default=1)

    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=64)
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

    if args.cuda >= 0:
        torch.cuda.set_device(args.cuda)

    # TODO(Alex): hid_dim depends on if pair input or not
    log.info("Loading tasks...")
    start_time = time.time()
    tasks, tok2idx = process_tasks(args.tasks.split(','), args.hid_dim * 4,
                                   args.max_vocab_size, args.max_seq_len,
                                   args.batch_size, args.cuda)
    log.info('\tFinished loading tasks in {0}s'.format(
        time.time() - start_time))

    log.info("Building model...")
    start_time = time.time()
    embeddings = load_embeddings(args.word_embs_file, tok2idx,
                                 args.word_dim)
    token_embedder = TokenEmbedder(embeddings, tok2idx)
    #token_embedder = nn.Embedding(args.max_vocab_size, args.word_dim,
    #                              padding_idx=0)

    encoder = MultiLayerRNNEncoder(args.word_dim, args.hid_dim,
                                   args.n_layers, nn.LSTMCell)
    model = MultiTaskModel(encoder, token_embedder, tasks)
    if args.cuda:
        model = model.cuda()
    log.info('\tFinished building model in {0}s'.format(
        time.time() - start_time))

    model.train_model(args.n_epochs, args.optimizer, args.lr)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
