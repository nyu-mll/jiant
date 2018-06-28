'''Preprocessing functions and pipeline

To add new tasks, add task-specific preprocessing functions to process_task()'''
import io
import os
import ipdb as pdb
import logging as log
from collections import defaultdict
import numpy as np
import torch

from allennlp.data import Instance, Vocabulary, Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer, \
    TokenCharactersIndexer
from allennlp_mods.numeric_field import NumericField

try:
    import fastText
except BaseException:
    log.info("fastText library not found!")

import _pickle as pkl

#  from serialize import write_records, read_records
import serialize

from tasks import SingleClassificationTask, PairClassificationTask, \
    PairRegressionTask, SequenceGenerationTask, RankingTask, \
    CoLATask, MRPCTask, MultiNLITask, MultiNLIFictionTask, \
    MultiNLISlateTask, MultiNLIGovernmentTask, MultiNLITravelTask, \
    MultiNLITelephoneTask, QQPTask, RTETask, \
    QNLITask, SNLITask, SSTTask, STSBTask, WNLITask, \
    LanguageModelingTask, PDTBTask, \
    WikiText2LMTask, WikiText103LMTask, DisSentBWBSingleTask, \
    DisSentWikiSingleTask, DisSentWikiFullTask, \
    JOCITask, PairOrdinalRegressionTask, WeakGroundedTask, \
    GroundedTask

NAME2INFO = {'sst': (SSTTask, 'SST-2/'),
             'cola': (CoLATask, 'CoLA/'),
             'mrpc': (MRPCTask, 'MRPC/'),
             'qqp': (QQPTask, 'QQP'),
             'sts-b': (STSBTask, 'STS-B/'),
             'mnli': (MultiNLITask, 'MNLI/'),
             'mnli-fiction': (MultiNLIFictionTask, 'MNLI/'),
             'mnli-slate': (MultiNLISlateTask, 'MNLI/'),
             'mnli-government': (MultiNLIGovernmentTask, 'MNLI/'),
             'mnli-telephone': (MultiNLITelephoneTask, 'MNLI/'),
             'mnli-travel': (MultiNLITravelTask, 'MNLI/'),
             'qnli': (QNLITask, 'QNLI/'),
             'rte': (RTETask, 'RTE/'),
             'snli': (SNLITask, 'SNLI/'),
             'wnli': (WNLITask, 'WNLI/'),
             'joci': (JOCITask, 'JOCI/'),
             'wiki2': (WikiText2LMTask, 'WikiText2/'),
             'wiki103': (WikiText103LMTask, 'WikiText103/'),
             'pdtb': (PDTBTask, 'PDTB/'),
             'dissentbwb': (DisSentBWBSingleTask, 'DisSent/bwb/'),
             'dissentwiki': (DisSentWikiSingleTask, 'DisSent/wikitext/'),
             'dissentwikifull': (DisSentWikiFullTask, 'DisSent/wikitext/'),
             'weakgrounded': (WeakGroundedTask, 'mscoco-temp/weakgrounded/'),
             'grounded': (GroundedTask, 'mscoco-temp/grounded/'),
             }

SOS_TOK, EOS_TOK = "<SOS>", "<EOS>"
SPECIALS = [SOS_TOK, EOS_TOK]

def _get_serialized_record_path(task_name, split, preproc_dir):
    serialized_record_path = os.path.join(preproc_dir, 
                                          "{:s}__{:s}_data".format(task_name, split))
    return serialized_record_path

def _get_instance_generator(task_name, split, preproc_dir):
    """Get a lazy generator for the given task and split.
    
    Args:
        task_name: (string), task name
        split: (string), split name ('train', 'val', or 'test')
        preproc_dir: (string) path to preprocessing dir

    Returns:
        serialize.RepeatableIterator yielding Instance objects
    """
    filename = _get_serialized_record_path(task_name, split, preproc_dir)
    return serialize.read_records(filename, repeatable=True)

def _serialize_task(task_name, split_dict, preproc_dir):
    """Serialize splits for the given task.

    Args:
        task_name: (string), task name
        split_dict: dict(string -> list(Instance))
        preproc_dir: (string) path to preprocessing dir
    """
    for split in split_dict:
        filename = _get_serialized_record_path(task_name, split, preproc_dir)
        serialize.write_records(split_dict[split], filename)

def del_field_tokens(instances):
    ''' Save memory by deleting the tokens that will no longer be used.
    
    Args:
        instances: iterable of Instance. Modified in-place.
    '''
    for instance in instances:
        if 'input1' in instance.fields:
            field = instance.fields['input1']
            del field.tokens
        if 'input2' in instance.fields:
            field = instance.fields['input2']
            del field.tokens

def build_tasks(args):
    '''Main logic for preparing tasks, doing so by
    1) creating / loading the tasks
    2) building / loading the vocabulary
    3) building / loading the word vectors
    4) indexing each task's data
    '''

    # 1) create / load tasks

    prepreproc_dir = os.path.join(args.exp_dir, "prepreproc")
    if not os.path.isdir(prepreproc_dir):
        os.mkdir(prepreproc_dir)
    tasks, train_task_names, eval_task_names = \
        get_tasks(args.train_tasks, args.eval_tasks, args.max_seq_len,
                  path=args.data_dir, scratch_path=args.exp_dir,
                  load_pkl=bool(not args.reload_tasks))

    # 2 + 3) build / load vocab and word vectors
    vocab_path = os.path.join(args.exp_dir, 'vocab')
    emb_file = os.path.join(args.exp_dir, 'embs.pkl')
    token_indexer = {}
    if not args.word_embs == 'none':
        token_indexer["words"] = SingleIdTokenIndexer()
    if args.elmo:
        token_indexer["elmo"] = ELMoTokenCharactersIndexer("elmo")
    if args.char_embs:
        token_indexer["chars"] = TokenCharactersIndexer("chars")
    if not args.reload_vocab and os.path.exists(vocab_path):
        vocab = Vocabulary.from_files(vocab_path)
        log.info("\tLoaded vocab from %s", vocab_path)
    else:
        log.info("\tBuilding vocab from scratch")
        max_v_sizes = {'word': args.max_word_v_size, 'char': args.max_char_v_size}
        word2freq, char2freq = get_words(tasks)
        vocab = get_vocab(word2freq, char2freq, max_v_sizes)
        vocab.save_to_files(vocab_path)
        log.info("\tSaved vocab to %s", vocab_path)
        del word2freq, char2freq
    word_v_size = vocab.get_vocab_size('tokens')
    char_v_size = vocab.get_vocab_size('chars')
    log.info("\tFinished building vocab. Using %d words, %d chars.",
             word_v_size, char_v_size)
    args.max_word_v_size, args.max_char_v_size = word_v_size, char_v_size
    if args.word_embs != 'none':
        if not args.reload_vocab and os.path.exists(emb_file):
            word_embs = pkl.load(open(emb_file, 'rb'))
        else:
            log.info("\tBuilding embeddings from scratch")
            if args.fastText:
                word_embs, _ = get_fastText_model(vocab, args.d_word,
                                                  model_file=args.fastText_model_file)
                log.info("\tNo pickling")
            else:
                word_embs = get_embeddings(vocab, args.word_embs_file, args.d_word)
                pkl.dump(word_embs, open(emb_file, 'wb'))
                log.info("\tSaved embeddings to %s", emb_file)
    else:
        word_embs = None

    # 4) Index tasks using vocab, using previous preprocessing if available.
    preproc_dir = os.path.join(args.exp_dir, "preproc")
    if not os.path.isdir(preproc_dir):
        os.mkdir(preproc_dir)
    preproc_file_names = []
    if not args.reload_vocab and not args.reload_indexing:
        for file in os.listdir(preproc_dir):
            preproc_file_names.append(file.split("__")[0])
        preproc_file_names = set(preproc_file_names)
    for task in tasks:
        if not task.name in preproc_file_names:
            log.info("\tIndexing task %s from scratch", task.name)
            split_dict = process_task(task, token_indexer, vocab)
            for instances in split_dict.values():
                del_field_tokens(instances)
            #  del_field_tokens(split_dict)
            _serialize_task(task.name, split_dict, preproc_dir)
            log.info("\tSaved data to %s", preproc_dir)

        # Replace lists of instances with lazy generators from disk.
        task.train_data = _get_instance_generator(task.name, "train", preproc_dir)
        task.val_data =   _get_instance_generator(task.name, "val", preproc_dir)
        task.test_data =  _get_instance_generator(task.name, "test", preproc_dir)
        log.info("\tLoaded indexed data for %s from %s", task.name, preproc_dir)

    log.info("\tFinished indexing tasks")

    train_tasks = [task for task in tasks if task.name in train_task_names]
    eval_tasks = [task for task in tasks if task.name in eval_task_names]
    log.info('\t  Training on %s', ', '.join(train_task_names))
    log.info('\t  Evaluating on %s', ', '.join(eval_task_names))
    return train_tasks, eval_tasks, vocab, word_embs


#  def serialize_instances_for_task(task, train_val_test_dict, preproc_dir):
#      for task_type in train_val_test_dict:
#          file_name = task.name + "__" + task_type
#          file_path = os.path.join(preproc_dir, file_name)
#          write_records(train_val_test_dict[task_type], file_path)
#          task_type_iterator = read_records(file_path)
#          setattr(task, task_type, task_type_iterator)


#  def get_task_generator(task_name, preproc_dir):
#      train_generator = read_records(os.path.join(preproc_dir, task_name + "__train_data"), repeatable=True)
#      val_generator = read_records(os.path.join(preproc_dir, task_name + "__val_data"),
#                                   repeatable=True)
#      test_generator = read_records(os.path.join(preproc_dir, task_name + "__test_data"))
#      return train_generator, val_generator, test_generator


def get_tasks(train_tasks, eval_tasks, max_seq_len, path=None,
              scratch_path=None, load_pkl=1):
    ''' Load tasks '''
    def parse_tasks(task_list):
        '''parse string of tasks'''
        if task_list == 'glue':
            tasks = ['sst', 'cola', 'mrpc', 'qqp', 'sts-b', 'mnli', 'qnli', 'rte', 'wnli']
        elif task_list == 'none':
            tasks = []
        else:
            tasks = task_list.split(',')
        return tasks

    train_task_names = parse_tasks(train_tasks)
    eval_task_names = parse_tasks(eval_tasks)
    task_names = list(set(train_task_names + eval_task_names))

    assert path is not None
    scratch_path = (scratch_path or path)
    log.info("Writing pre-preprocessed tasks to %s", scratch_path)

    tasks = []
    for name in task_names:
        assert name in NAME2INFO, 'Task not found!'
        task_src_path = os.path.join(path, NAME2INFO[name][1])
        task_scratch_path = os.path.join(scratch_path, NAME2INFO[name][1])
        pkl_path = os.path.join(task_scratch_path, "%s_task.pkl" % name)
        if os.path.isfile(pkl_path) and load_pkl:
            task = pkl.load(open(pkl_path, 'rb'))
            log.info('\tLoaded existing task %s', name)
        else:
            log.info('\tCreating task %s from scratch', name)
            task = NAME2INFO[name][0](task_src_path, max_seq_len, name)
            if not os.path.isdir(task_scratch_path):
                os.mkdir(task_scratch_path)
            pkl.dump(task, open(pkl_path, 'wb'))
        #task.truncate(max_seq_len, SOS_TOK, EOS_TOK)
        tasks.append(task)

    for task in tasks: # hacky
        task.n_tr_examples = len(task.train_data_text[0])
        task.n_val_examples = len(task.val_data_text[0])
        task.n_te_examples = len(task.test_data_text[0])

    log.info("\tFinished loading tasks: %s.", ' '.join([task.name for task in tasks]))
    return tasks, train_task_names, eval_task_names


def get_words(tasks):
    '''
    Get all words for all tasks for all splits for all sentences
    Return dictionary mapping words to frequencies.
    '''
    word2freq, char2freq = defaultdict(int), defaultdict(int)

    def count_sentence(sentence):
        '''Update counts for words in the sentence'''
        for word in sentence:
            word2freq[word] += 1
            for char in list(word):
                char2freq[char] += 1
        return

    for task in tasks:
        for sentence in task.sentences:
            count_sentence(sentence)

    log.info("\tFinished counting words")
    return word2freq, char2freq


def get_vocab(word2freq, char2freq, max_v_sizes):
    '''Build vocabulary'''
    vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes)
    for special in SPECIALS:
        vocab.add_token_to_namespace(special, 'tokens')

    words_by_freq = [(word, freq) for word, freq in word2freq.items()]
    words_by_freq.sort(key=lambda x: x[1], reverse=True)
    for word, _ in words_by_freq[:max_v_sizes['word']]:
        vocab.add_token_to_namespace(word, 'tokens')

    chars_by_freq = [(char, freq) for char, freq in char2freq.items()]
    chars_by_freq.sort(key=lambda x: x[1], reverse=True)
    for char, _ in chars_by_freq[:max_v_sizes['char']]:
        vocab.add_token_to_namespace(char, 'chars')
    return vocab


#  def del_field_tokens(train_val_test_dict):
#      ''' Save memory by deleting the tokens that will no longer be used '''
#      all_instances = []
#      for task_type in train_val_test_dict:
#          all_instances += train_val_test_dict[task_type]
#      for instance in all_instances:
#          if 'input1' in instance.fields:
#              field = instance.fields['input1']
#              del field.tokens
#          if 'input2' in instance.fields:
#              field = instance.fields['input2']
#              del field.tokens


def get_embeddings(vocab, vec_file, d_word):
    '''Get embeddings for the words in vocab'''
    word_v_size, unk_idx = vocab.get_vocab_size('tokens'), vocab.get_token_index(vocab._oov_token)
    embeddings = np.random.randn(word_v_size, d_word)
    with io.open(vec_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as vec_fh:
        for line in vec_fh:
            word, vec = line.split(' ', 1)
            idx = vocab.get_token_index(word)
            if idx != unk_idx:
                embeddings[idx] = np.array(list(map(float, vec.split())))
    embeddings[vocab.get_token_index(vocab._padding_token)] = 0.
    embeddings = torch.FloatTensor(embeddings)
    log.info("\tFinished loading embeddings")
    return embeddings


def get_fastText_model(vocab, d_word, model_file=None):
    '''
    Same interface as get_embeddings except for fastText. Note that if the path to the model
    is provided, the embeddings will rely on that model instead.
    **Crucially, the embeddings from the pretrained model DO NOT match those from the released
    vector file**
    '''
    word_v_size, unk_idx = vocab.get_vocab_size('tokens'), vocab.get_token_index(vocab._oov_token)
    embeddings = np.random.randn(word_v_size, d_word)
    model = fastText.FastText.load_model(model_file)
    special_tokens = [vocab._padding_token, vocab._oov_token]
    # We can also just check if idx >= 2
    for idx in range(word_v_size):
        word = vocab.get_token_from_index(idx)
        if word in special_tokens:
            continue
        embeddings[idx] = model.get_word_vector(word)
    embeddings[vocab.get_token_index(vocab._padding_token)] = 0.
    embeddings = torch.FloatTensor(embeddings)
    log.info("\tFinished loading pretrained fastText model and embeddings")
    return embeddings, model


def process_task(task, token_indexer, vocab):
    '''
    Convert a task's splits into AllenNLP fields then index the splits using vocab.
    Different tasks have different formats and fields, so process_task routes tasks
    to the corresponding processing based on the task type. These task specific processing
    functions should return three splits, which are lists (possibly empty) of AllenNLP instances.
    These instances are then indexed using the vocab
    '''
    split_dict = {}
    for split_name in ['train', 'val', 'test']:
        split_text = getattr(task, '%s_data_text' % split_name)
        if isinstance(task, SingleClassificationTask):
            split = process_single_pair_task_split(split_text, token_indexer, is_pair=False)
        elif isinstance(task, PairClassificationTask):
            split = process_single_pair_task_split(split_text, token_indexer, is_pair=True)
        elif isinstance(task, PairRegressionTask):
            split = process_single_pair_task_split(split_text, token_indexer, is_pair=True,
                                                   classification=False)
        elif isinstance(task, PairOrdinalRegressionTask):
            split = process_single_pair_task_split(split_text, token_indexer, is_pair=True,
                                                   classification=False)
        elif isinstance(task, LanguageModelingTask):
            split = process_lm_task_split(split_text, token_indexer)
        elif isinstance(task, SequenceGenerationTask):
            pass
        elif isinstance(task, GroundedTask):
            split = process_grounded_task_split(split_text, token_indexer, is_pair=False, classification=True)
        elif isinstance(task, RankingTask):
            pass
        else:
            raise ValueError("Preprocessing procedure not found for %s" % task.name)
        for instance in split:
            instance.index_fields(vocab)
        split_dict[split_name] = split
    return split_dict

def process_grounded_task_split(split, indexers, is_pair=True, classification=True):
    '''
    Convert a dataset of sentences into padded sequences of indices.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)

    Returns:
    '''
    inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
    labels = [NumericField(l) for l in split[1]]
    ids = [NumericField(l) for l in split[2]]
    instances = [Instance({"input1": input1, "labels": label, "ids": ids}) for (input1, label, ids) in
                         zip(inputs1, labels, ids)]
    
    return instances  # DatasetReader(instances) #Batch(instances) #Dataset(instances)


def process_single_pair_task_split(split, indexers, is_pair=True, classification=True):
    '''
    Convert a dataset of sentences into padded sequences of indices.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)

    Returns:
    '''
    if is_pair:
        inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        inputs2 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[1]]
        if classification:
            labels = [LabelField(l, label_namespace="labels", skip_indexing=True) for l in split[2]]
        else:
            labels = [NumericField(l) for l in split[-1]]

        if len(split) == 4:  # numbered test examples
            idxs = [LabelField(l, label_namespace="idxs", skip_indexing=True) for l in split[3]]
            instances = [Instance({"input1": input1, "input2": input2, "labels": label, "idx": idx})
                         for (input1, input2, label, idx) in zip(inputs1, inputs2, labels, idxs)]

        else:
            instances = [Instance({"input1": input1, "input2": input2, "labels": label}) for
                         (input1, input2, label) in zip(inputs1, inputs2, labels)]

    else:
        inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        if classification:
            labels = [LabelField(l, label_namespace="labels", skip_indexing=True) for l in split[2]]
        else:
            labels = [NumericField(l) for l in split[2]]

        if len(split) == 4:
            idxs = [LabelField(l, label_namespace="idxs", skip_indexing=True) for l in split[3]]
            instances = [Instance({"input1": input1, "labels": label, "idx": idx}) for
                         (input1, label, idx) in zip(inputs1, labels, idxs)]
        else:
            instances = [Instance({"input1": input1, "labels": label}) for (input1, label) in
                         zip(inputs1, labels)]
    return instances  # DatasetReader(instances) #Batch(instances) #Dataset(instances)


def process_lm_task_split(split, indexers):
    ''' Process a language modeling split '''
    inp_fwd = [TextField(list(map(Token, sent[:-1])), token_indexers=indexers) for sent in split]
    inp_bwd = [TextField(list(map(Token, sent[::-1][:-1])), token_indexers=indexers)
               for sent in split]
    trg_fwd = [TextField(list(map(Token, sent[1:])), token_indexers=indexers) for sent in split]
    trg_bwd = [TextField(list(map(Token, sent[::-1][1:])), token_indexers=indexers)
               for sent in split]
    # instances = [Instance({"input": inp, "targs": trg_f, "targs_b": trg_b})
    #             for (inp, trg_f, trg_b) in zip(inputs, trg_fwd, trg_bwd)]
    instances = [Instance({"input": inp_f, "input_bwd": inp_b, "targs": trg_f, "targs_b": trg_b})
                 for (inp_f, inp_b, trg_f, trg_b) in zip(inp_fwd, inp_bwd, trg_fwd, trg_bwd)]
    return instances
