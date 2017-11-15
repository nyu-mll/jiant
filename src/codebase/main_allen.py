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

#from allennlp.commands.evaluate import evaluate
from allennlp.common.params import Params
from allennlp.data import Instance, Dataset, Vocabulary
from allennlp.data.fields import TextField, LabelField
from allennlp.data.iterators import DataIterator, BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.similarity_functions import DotProductSimilarity
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, \
                                              CnnEncoder
from allennlp.modules.seq2seq_encoders import IntraSentenceAttentionEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
from allennlp.training import Trainer

PATH_TO_PKG = '../'
sys.path.append(os.path.join(os.path.dirname(__file__), PATH_TO_PKG))
from codebase.tasks import Task, MSRPTask, QuoraTask, SNLITask, SSTTask
from codebase.models_allen import HeadlessBiDAF, HeadlessSentEncoder, \
                                    MultiTaskModel
from codebase.trainer import MultiTaskTrainer
from codebase.evaluate import evaluate
from codebase.utils.encoders import MultiLayerRNNEncoder
from codebase.utils.seq_batch import SequenceBatch
from codebase.utils.token_embedder import TokenEmbedder
from codebase.utils.utils import GPUVariable

PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/processed_data/' + \
              'mtl-sentence-representations/'

NAME2TASK = {'msrp': MSRPTask,
             'quora': QuoraTask, 'snli': SNLITask,
             'sst': SSTTask,
             'small':QuoraTask, 'small2': QuoraTask}
NAME2DATA = {'msrp': PATH_PREFIX + 'MRPC/',
             'quora': PATH_PREFIX + 'Quora/quora_duplicate_questions.tsv',
             'snli': PATH_PREFIX + 'SNLI/',
             'sst': PATH_PREFIX + 'SST/binary/',
             'small': PATH_PREFIX + 'Quora/quora_small.tsv',
             'small2': PATH_PREFIX + 'Quora/quora_small.tsv'}

# lazy way to map tasks to vocab pickles
NAME2SAVE = {'msrp': PATH_PREFIX + 'MRPC/msrp_task.pkl',
             'quora': PATH_PREFIX + 'Quora/quora_task.pkl',
             'snli': PATH_PREFIX + 'SNLI/snli_task.pkl',
             'sst': PATH_PREFIX + 'SST/sst_task.pkl',
             'small': PATH_PREFIX + 'Quora/small_task.pkl',
             'small2': PATH_PREFIX + 'Quora/small2_task.pkl'}

def process_tasks(task_names, input_dim, max_vocab_size, max_seq_len,
                  vocab_path, cuda, load_tasks, load_vocab):
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
        if os.path.isfile(NAME2SAVE[name]) and load_tasks:
            tasks.append(pkl.load(open(NAME2SAVE[name], 'rb')))
            log.info('\t\tLoaded existing task %s', tasks[-1].name)
        else:
            task = (NAME2TASK[name](NAME2DATA[name], input_dim,
                                    max_seq_len, cuda, name))
            tasks.append(task)
            pkl.dump(task, open(NAME2SAVE[name], 'wb'))
    log.info("\tFinished loading tasks: %s.", ' '.join(
        [task.name for task in tasks]))

    token_indexer = {"words": SingleIdTokenIndexer(),
                     "chars": TokenCharactersIndexer()}

    for task in tasks:
        task.train_data = process_split(task.train_data_text,
                                        task.pair_input, token_indexer)
        task.val_data = process_split(task.val_data_text, task.pair_input,
                                      token_indexer)
        task.test_data = process_split(task.test_data_text, task.pair_input,
                                       token_indexer)

    # assuming one task for now
    # build vocabulary
    if load_vocab and os.path.exists(vocab_path + 'non_padded_namespaces.txt'):
        vocab = Vocabulary.from_files(vocab_path)
        log.info('\t\tLoaded existing vocabulary from %s', vocab_path)
    else:
        vocab = Vocabulary.from_dataset(tasks[0].train_data,
                                        max_vocab_size=max_vocab_size)
        if vocab_path:
            vocab.save_to_files(vocab_path)
            log.info('\t\tSaved vocabulary to %s', vocab_path)
    log.info("\tFinished building vocab.")

    # index words and chars using vocab
    for task in tasks:
        task.train_data.index_instances(vocab)
        task.val_data.index_instances(vocab)
        task.test_data.index_instances(vocab)
    log.info("\tFinished indexing.")

    return tasks, vocab

def process_split(split, pair_input, token_indexer):
    '''
    Convert a dataset of sentences into padded sequences of indices.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)

    Returns:
    '''
    if pair_input:
        #inputs1 = [TextField(sent, token_indexers={"words": SingleIdTokenIndexer(), "chars": TokenCharactersIndexer()}) for sent in split[0]]
        #inputs2 = [TextField(sent, token_indexers={"words": SingleIdTokenIndexer(), "chars": TokenCharactersIndexer()}) for sent in split[1]]
        inputs1 = [TextField(sent, token_indexers=token_indexer) for \
                   sent in split[0]]
        inputs2 = [TextField(sent, token_indexers=token_indexer) for \
                   sent in split[1]]
        labels = [LabelField(l, label_namespace="labels",
                             skip_indexing=True) for l in split[-1]]
        instances = [Instance({"input1": input1, "input2": input2, \
                     "label": label}) for (input1, input2, label) in
                     zip(inputs1, inputs2, labels)]

    else:
        inputs1 = [TextField(sent, token_indexers=token_indexer) for \
                   sent in split[0]]
        labels = [LabelField(l, label_namespace="labels",
                             skip_indexing=True) for l in split[-1]]
        instances = [Instance({"input1": input1, \
                     "label": label}) for (input1, label) in
                     zip(inputs1, labels)]
    return Dataset(instances)

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

def load_word_embeddings(path, vocab, word_dim):
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
    vocab_size = vocab.get_vocab_size('tokens')
    unk_idx = vocab.get_token_index('@@UNKNOWN@@')
    pad_idx = vocab.get_token_index('@@PADDING@@')
    embeddings = np.random.rand(vocab.get_vocab_size('tokens'), 
                                word_dim).astype(float)
    n_embs = 0
    with open(path) as fh:
        for row in fh:
            row = row.split()
            word = row[0]
            idx = vocab.get_token_index(word)
            if idx != unk_idx:
                assert len(row[1:]) == word_dim
                embeddings[idx] = np.array([float(v) for v in row[1:]])
                n_embs += 1

    embeddings[pad_idx] = 0.
    log.info("\tLoaded pretrained embeddings for %d/%d words",
             n_embs, vocab_size)
    return torch.FloatTensor(embeddings)

def main(arguments):
    '''
    Train or load a model. Evaluate on some tasks.
    '''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--cuda', help='0 if no CUDA, else gpu id',
                        type=int, default=0)

    parser.add_argument('--log_file', help='path to log to',
                        type=str, default=0)
    parser.add_argument('--save_dir', help='path to log to',
                        type=str, default='')
    parser.add_argument('--data_path', help='path to directory containing '
                        ' {train,val,test}.pkl', type=str, default='')
    parser.add_argument('--vocab_path', help='path to directory containing '
                        ' {train,val,test}.pkl', type=str, default='')
    parser.add_argument('--word_embs_file', help='file containing word ' +
                        'embeddings', type=str, default='')

    parser.add_argument('--load_model', help='1 if load from checkpoint',
                        type=int, default=1)
    parser.add_argument('--load_tasks', help='1 if load tasks',
                        type=int, default=1)
    parser.add_argument('--load_vocab', help='1 if load vocabulary',
                        type=int, default=1)

    parser.add_argument('--tasks', help='comma separated list of tasks',
                        type=str, default='')
    parser.add_argument('--max_vocab_size', help='vocabulary size',
                        type=int, default=10000)
    parser.add_argument('--max_seq_len', help='max sequence length',
                        type=int, default=40)

    parser.add_argument('--max_char_vocab_size', help='char vocabulary size',
                        type=int, default=2000)
    parser.add_argument('--n_char_filters', help='number of conv filters for '+
                        'char embedding combiner', type=int, default=64)
    parser.add_argument('--char_filter_sizes', help='filter sizes for char ' +
                        'embedding combiner', type=str, default='2,3,4,5')

    parser.add_argument('--word_dim', help='dimension of word embeddings',
                        type=int, default=300)
    parser.add_argument('--char_dim', help='dimension of char embeddings',
                        type=int, default=100)
    parser.add_argument('--hid_dim', help='hidden dimension size',
                        type=int, default=4096)
    parser.add_argument('--n_layers', help='number of RNN layers',
                        type=int, default=1)
    parser.add_argument('--n_highway_layers', help='number of highway layers',
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

    log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO,
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = log.FileHandler(args.log_file)
    log.getLogger().addHandler(file_handler)
    log.info(args)

    if args.cuda >= 0:
        log.info("Using GPU %d", args.cuda)
        torch.cuda.set_device(args.cuda)

    log.info("Loading tasks...")
    start_time = time.time()
    dim = args.word_dim + args.char_dim
    tasks, vocab = process_tasks(args.tasks.split(','), dim,
                                 args.max_vocab_size, args.max_seq_len,
                                 args.vocab_path, args.cuda,
                                 args.load_tasks, args.load_vocab)
    log.info('\tFinished loading tasks in %.3fs', time.time() - start_time)

    start_time = time.time()
    word_dim, char_dim = args.word_dim, args.char_dim
    dim = word_dim + char_dim
    if args.word_embs_file:
        word_embs = load_word_embeddings(args.word_embs_file, vocab, word_dim)
    else:
        word_embs = None

    word_embedder = Embedding(vocab.get_vocab_size('tokens'), word_dim,
                              weight=word_embs)
    char_embedder = Embedding(vocab.get_vocab_size('token_characters'),
                              char_dim)
    # TODO(Alex): should be CNN
    #char_encoder = BagOfEmbeddingsEncoder(char_dim, True)
    filter_sizes = tuple([int(i) for i in args.char_filter_sizes.split(',')])
    char_encoder = CnnEncoder(args.char_dim, num_filters=args.n_char_filters,
                              ngram_filter_sizes=filter_sizes,
                              output_dim=args.char_dim)
    char_embedder = TokenCharactersEncoder(char_embedder, char_encoder)

    token_embedder = {"words": word_embedder, "chars": char_embedder}
    text_field_embedder = BasicTextFieldEmbedder(token_embedder)
    phrase_layer = s2s_e.by_name('lstm').from_params(Params({
        'input_size': dim,
        'hidden_size': 2 * dim,
        'bidirectional': True}))
    modeling_layer = s2s_e.by_name('lstm').from_params(Params({
        'input_size': 16 * dim,
        'hidden_size': 2 * dim,
        'bidirectional': True}))
    sent_encoder = HeadlessSentEncoder(vocab, text_field_embedder,
                                       args.n_highway_layers,
                                       phrase_layer)
    pair_encoder = HeadlessBiDAF(vocab, text_field_embedder,
                                 args.n_highway_layers,
                                 phrase_layer,
                                 DotProductSimilarity(),
                                 modeling_layer)
    model = MultiTaskModel(sent_encoder, pair_encoder)
    if args.cuda >= 0:
        model = model.cuda()
    log.info('\tFinished building model in %.3fs', time.time() - start_time)

    # Set up Trainer and train
    optimizer_params = Params({'type':'sgd', 'lr':args.lr})
    train_params = Params({'num_epochs':args.n_epochs,
                           'cuda_device':args.cuda,
                           'optimizer':optimizer_params,
                           'validation_metric':'-%s_loss' % tasks[0].name,
                           'no_tqdm': True})
    iterator = BasicIterator(args.batch_size)
    trainer = MultiTaskTrainer.from_params(model, args.save_dir, iterator,
                                           train_params)

    #trainer.train()
    trainer.train(tasks, args.load_model)

    # Evaluate
    results = evaluate(model, tasks, iterator, cuda_device=args.cuda)
    log.info('*** TEST RESULTS ***')
    for name, value in results.items():
        log.info("%s\t%3f", name, value)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
