'''Preprocessing functions and pipeline'''
import os
import pdb # pylint disable=unused-import
import logging as log
from collections import defaultdict
import _pickle as pkl
import numpy as np
import torch

from allennlp.data import Instance, Dataset, Vocabulary, Token
from allennlp.data.fields import TextField, LabelField, NumericField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, \
                                         ELMoTokenCharactersIndexer

from tasks import MSRPTask, MultiNLITask, QuoraTask, RTE8Task, STS14Task, \
                  RTETask, RTE5Task, SQuADTask, SNLITask, SSTTask, \
                  STSBenchmarkTask, TwitterIronyTask

PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/processed_data/mtl-sentence-representations/'
#PATH_PREFIX = '/beegfs/aw3272/processed_data/mtl-sentence-representations/'


ALL_TASKS = ['mnli', 'msrp', 'quora', 'rte', 'squad', 'snli', 'sst', 'sts-benchmark']
NAME2TASK = {'msrp': MSRPTask, 'mnli': MultiNLITask,
             'quora': QuoraTask, 'rte8': RTE8Task,
             'rte': RTETask, 'rte5': RTE5Task,
             'squad': SQuADTask, 'snli': SNLITask,
             'sst': SSTTask, 'sts14': STS14Task,
             'sts-benchmark': STSBenchmarkTask,
             'small': QuoraTask, 'small2': QuoraTask,
             'twitter-irony': TwitterIronyTask}
NAME2DATA = {'msrp': PATH_PREFIX + 'MRPC/',
             'mnli': PATH_PREFIX + 'MNLI/',
             'quora': PATH_PREFIX + 'Quora/quora_duplicate_questions.tsv',
             'rte': PATH_PREFIX + 'rte/',
             'rte5': PATH_PREFIX + 'rte/',
             'rte8': PATH_PREFIX + 'rte8/semeval2013-Task7-2and3way',
             'squad': PATH_PREFIX + 'squad/',
             'snli': PATH_PREFIX + 'SNLI/',
             'sst': PATH_PREFIX + 'SST/binary/',
             'sts14': PATH_PREFIX + 'STS/STS14-en-test/',
             'sts-benchmark': PATH_PREFIX + 'STS/STSBenchmark/',
             'small': PATH_PREFIX + 'Quora/quora_small.tsv',
             'small2': PATH_PREFIX + 'Quora/quora_small.tsv'}

# lazy way to map tasks to preprocessed tasks
NAME2SAVE = {'msrp': PATH_PREFIX + 'MRPC/',
             'mnli': PATH_PREFIX + 'MNLI/',
             'quora': PATH_PREFIX + 'Quora/',
             'rte': PATH_PREFIX + 'rte/',
             'rte5': PATH_PREFIX + 'rte/',
             'rte8': PATH_PREFIX + 'rte8/',
             'squad': PATH_PREFIX + 'squad/',
             'snli': PATH_PREFIX + 'SNLI/',
             'sst': PATH_PREFIX + 'SST/',
             'sts14': PATH_PREFIX + 'STS/',
             'sts-benchmark': PATH_PREFIX + 'STS/',
             'small': PATH_PREFIX + 'Quora/',
             'small2': PATH_PREFIX + 'Quora/',
             'twitter-irony': PATH_PREFIX + 'twitter-irony/'}

def build_tasks(args):
    '''Prepare tasks'''

    def parse_tasks(task_list):
        '''parse string of tasks'''
        if task_list == 'all':
            tasks = ALL_TASKS
        elif task_list == 'none':
            tasks = []
        else:
            tasks = task_list.split(',')
        return tasks

    train_task_names = parse_tasks(args.train_tasks)
    eval_task_names = parse_tasks(args.eval_tasks)
    all_task_names = list(set(train_task_names + eval_task_names))
    tasks = get_tasks(all_task_names, args.max_seq_len, args.load_tasks)

    max_v_sizes = {'word': args.max_word_v_size, 'char': args.max_char_v_size}
    token_indexer = {"words": SingleIdTokenIndexer(), "chars": TokenCharactersIndexer("chars")}
    if args.elmo:
        token_indexer["elmo"] = ELMoTokenCharactersIndexer("elmo")

    vocab_path = os.path.join(args.exp_dir, 'vocab')
    preproc_file = os.path.join(args.exp_dir, args.preproc_file)
    if args.load_preproc and os.path.exists(args.preproc_file):
        preproc = pkl.load(open(preproc_file, 'rb'))
        vocab = Vocabulary.from_files(vocab_path)
        word_embs = preproc['word_embs']
        for task in tasks:
            train, val, test = preproc[task.name]
            task.train_data = train
            task.val_data = val
            task.test_data = test
        log.info("\tFinished building vocab. Using %d words, %d chars",
                 vocab.get_vocab_size('tokens'), vocab.get_vocab_size('chars'))
        log.info("\tLoaded data from %s", preproc_file)

    else:
        log.info("\tProcessing tasks from scratch")
        word2freq, char2freq = get_words(tasks)
        word2vec = get_word_vecs(args.word_embs_file, word2freq)
        vocab = get_vocab(word2vec, word2freq, char2freq, max_v_sizes)
        word_embs = get_embeddings(vocab, word2vec, args.d_word)
        preproc = {'word_embs': word_embs}
        for task in tasks:
            train, val, test = process_task(task, token_indexer, vocab)
            task.train_data = train
            task.val_data = val
            task.test_data = test
            preproc[task.name] = (train, val, test)
        log.info("\tFinished indexing tasks")
        pkl.dump(preproc, open(preproc_file, 'wb'))
        vocab.save_to_files(vocab_path)
        log.info("\tSaved data to %s", args.exp_dir)


    train_tasks = [task for task in tasks if task.name in train_task_names]
    eval_tasks = [task for task in tasks if task.name in eval_task_names]
    return train_tasks, eval_tasks, vocab, word_embs

def get_tasks(task_names, max_seq_len, load):
    '''
    Load tasks
    '''
    tasks = []
    for name in task_names:
        assert name in NAME2TASK, 'Task not found!'
        pkl_path = NAME2SAVE[name] + "%s_task.pkl" % name
        if os.path.isfile(pkl_path) and load:
            task = pkl.load(open(pkl_path, 'rb'))
            log.info('\tLoaded existing task %s', name)
        else:
            task = NAME2TASK[name](NAME2DATA[name], max_seq_len, name)
            pkl.dump(task, open(pkl_path, 'wb'))
        tasks.append(task)
    log.info("\tFinished loading tasks: %s.", ' '.join([task.name for task in tasks]))
    return tasks

def get_words(tasks):
    '''
    Get all words for all tasks for all splits for all sentences
    Return dictionary mapping words to frequencies.
    '''
    word2freq = defaultdict(int)
    char2freq = defaultdict(int)
    def count_sentence(sentence):
        '''Update counts for words and chars in the sentence'''
        for word in sentence:
            word2freq[word] += 1
            for char in list(word):
                char2freq[char] += 1
        return

    for task in tasks:
        for split in [task.train_data_text, task.val_data_text, task.test_data_text]:
            for sentence in split[0]:
                count_sentence(sentence)
            if task.pair_input:
                for sentence in split[1]:
                    count_sentence(sentence)
    log.info("\tFinished counting words and chars")
    return word2freq, char2freq

def get_word_vecs(vec_file, vocab):
    '''Load all vectors for all words in vocab with vectors'''
    word_vecs = {}
    with open(vec_file) as vec_fh:
        for line in vec_fh:
            word, vec = line.split(' ', 1)
            if word in vocab:
                word_vecs[word] = np.array(list(map(float, vec.split())))
    log.info('\tFinished retrieving word vectors. Found %d/%d words with word vectors',
             len(word_vecs), len(vocab))
    return word_vecs

def get_vocab(word2vec, word2freq, char2freq, max_v_sizes):
    '''Build vocabulary'''
    vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes['word'])
    words_by_freq = [(word, freq) for word, freq in word2freq.items() if
                     word in word2vec]
    words_by_freq.sort(key=lambda x: x[1], reverse=True)
    chars_by_freq = [(c, freq) for c, freq in char2freq.items()]
    chars_by_freq.sort(key=lambda x: x[1], reverse=True)
    for word, _ in words_by_freq[:max_v_sizes['word']]:
        vocab.add_token_to_namespace(word, 'tokens')
    for char, _ in chars_by_freq[:max_v_sizes['char']]:
        vocab.add_token_to_namespace(char, 'chars')
    log.info("\tFinished building vocab. Using %d words, %d chars", vocab.get_vocab_size('tokens'),
             vocab.get_vocab_size('chars'))
    return vocab

def get_embeddings(vocab, word2vec, d_word):
    '''Get embeddings for the words in vocab'''
    word_v_size = vocab.get_vocab_size('tokens')
    embeddings = np.zeros((word_v_size, d_word))
    for idx in range(word_v_size): # kind of hacky
        word = vocab.get_token_from_index(idx)
        if word == '@@PADDING@@' or word == '@@UNKNOWN@@':
            continue
        try:
            assert word in word2vec
        except AssertionError as error:
            log.debug(error)
            pdb.set_trace()
        embeddings[idx] = word2vec[word]
    embeddings[vocab.get_token_index('@@PADDING@@')] = 0.
    embeddings = torch.FloatTensor(embeddings)
    log.info("\tFinished loading embeddings")
    return embeddings

def process_task(task, token_indexer, vocab):
    '''
    Convert a task's splits into AllenNLP fields then
    Index the splits using the given vocab (experiment dependent)
    '''
    train = process_split(task.train_data_text, token_indexer, task.pair_input, task.categorical)
    val = process_split(task.val_data_text, token_indexer, task.pair_input, task.categorical)
    test = process_split(task.test_data_text, token_indexer, task.pair_input, task.categorical)
    train.index_instances(vocab)
    val.index_instances(vocab)
    test.index_instances(vocab)
    return train, val, test

def process_split(split, indexers, pair_input, categorical):
    '''
    Convert a dataset of sentences into padded sequences of indices.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)

    Returns:
    '''
    if pair_input:
        inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        inputs2 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[1]]
        if categorical:
            labels = [LabelField(l, label_namespace="labels", skip_indexing=True) for l in split[-1]]
        else:
            labels = [NumericField(l) for l in split[-1]]

        instances = [Instance({"input1": input1, "input2": input2, "label": label}) for \
                        (input1, input2, label) in zip(inputs1, inputs2, labels)]

    else:
        inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        if categorical:
            labels = [LabelField(l, label_namespace="labels", skip_indexing=True) for l in split[-1]]
        else:
            labels = [NumericField(l) for l in split[-1]]

        instances = [Instance({"input1": input1, "label": label}) for (input1, label) in
                     zip(inputs1, labels)]
    return Dataset(instances)

