'''Preprocessing functions and pipeline

The pipeline is three steps
    1) create / load tasks, which includes
        a) load raw data
        b) tokenize raw data
    2) create / load all vocabularies (word, char, task-specific target vocabs)
        a) count tokens of a vocab
        b) take the N most frequent tokens
    3) index all the data using appropriate indexers
        We save indexed data to streamable Records to save memory.
'''
import io
import os
import sys
import copy
import logging as log
from collections import defaultdict
import numpy as np
import torch

from allennlp.data import Vocabulary
from allennlp.data.token_indexers import \
    SingleIdTokenIndexer, ELMoTokenCharactersIndexer, \
    TokenCharactersIndexer

try:
    import fastText
except BaseException:
    log.info("fastText library not found!")

import _pickle as pkl  #  :(

from . import config
from . import serialize
from . import utils
from . import tasks as tasks_module

from .tasks import \
    CoLATask, MRPCTask, MultiNLITask, QQPTask, QQPAltTask, RTETask, \
    QNLITask, QNLIAltTask, SNLITask, SSTTask, STSBTask, STSBAltTask, WNLITask, \
    PDTBTask, \
    WikiText2LMTask, WikiText103LMTask, DisSentBWBSingleTask, \
    DisSentWikiSingleTask, DisSentWikiBigFullTask, \
    JOCITask, PairOrdinalRegressionTask, WeakGroundedTask, \
    GroundedTask, MTTask, BWBLMTask, WikiInsertionsTask, \
    NLITypeProbingTask, MultiNLIAltTask, VAETask, \
    GroundedSWTask, NLITypeProbingAltTask

from .tasks import \
    RecastKGTask, RecastLexicosynTask, RecastWinogenderTask, \
    RecastFactualityTask, RecastSentimentTask, RecastVerbcornerTask, \
    RecastVerbnetTask, RecastNERTask, RecastPunTask, TaggingTask, \
    MultiNLIFictionTask, MultiNLISlateTask, MultiNLIGovernmentTask, \
    MultiNLITravelTask, MultiNLITelephoneTask, NPSTask
from .tasks import CCGTaggingTask, MultiNLIDiagnosticTask


ALL_GLUE_TASKS = ['sst', 'cola', 'mrpc', 'qqp', 'sts-b',
                  'mnli', 'qnli', 'rte', 'wnli', 'mnli-diagnostic']

# people are mostly using nli-prob for now, but we will change to
# using individual tasks later, so better to have as a list
ALL_NLI_PROBING_TASKS = ['nli-prob', 'nps', 'nli-prob-prepswap', 'nli-prob-negation', 'nli-alt']

# Edge probing suite.
ALL_EDGE_TASKS = ['edges-srl-conll2005', 'edges-spr2',
                  'edges-dpr', 'edges-ner-conll2003',
                  'edges-coref-ontonotes',
                  'edges-dep-labeling']

# Tasks for which we need to construct task-specific vocabularies
ALL_TARG_VOC_TASKS = ['wmt17_en_ru', 'wmt14_en_de', 'reddit_s2s',
                    'reddit_s2s_3.4G', 'reddit_s2s_dummy', 'wiki103_s2s']

# DEPRECATED: use @register_task in tasks.py instead.
NAME2INFO = {'sst': (SSTTask, 'SST-2/'),
             'cola': (CoLATask, 'CoLA/'),
             'mrpc': (MRPCTask, 'MRPC/'),
             'qqp': (QQPTask, 'QQP'),
             'qqp-alt': (QQPAltTask, 'QQP'),
             'sts-b': (STSBTask, 'STS-B/'),
             'sts-b-alt': (STSBAltTask, 'STS-B/'),
             'mnli': (MultiNLITask, 'MNLI/'),
             'mnli-alt': (MultiNLIAltTask, 'MNLI/'),
             'mnli-fiction': (MultiNLIFictionTask, 'MNLI/'),
             'mnli-slate': (MultiNLISlateTask, 'MNLI/'),
             'mnli-government': (MultiNLIGovernmentTask, 'MNLI/'),
             'mnli-telephone': (MultiNLITelephoneTask, 'MNLI/'),
             'mnli-travel': (MultiNLITravelTask, 'MNLI/'),
             'mnli-diagnostic': (MultiNLIDiagnosticTask, 'MNLI/'),
             'qnli': (QNLITask, 'QNLI/'),
             'qnli-alt': (QNLIAltTask, 'QNLI/'),
             'rte': (RTETask, 'RTE/'),
             'snli': (SNLITask, 'SNLI/'),
             'wnli': (WNLITask, 'WNLI/'),
             'joci': (JOCITask, 'JOCI/'),
             'wiki2': (WikiText2LMTask, 'WikiText2/'),
             'wiki103': (WikiText103LMTask, 'WikiText103/'),
             'bwb': (BWBLMTask, 'BWB/'),
             'pdtb': (PDTBTask, 'PDTB/'),
             'wikiins': (WikiInsertionsTask, 'wiki-insertions'),
             'dissentbwb': (DisSentBWBSingleTask, 'DisSent/bwb/'),
             'dissentwiki': (DisSentWikiSingleTask, 'DisSent/wikitext/'),
             'dissentwikifullbig': (DisSentWikiBigFullTask, 'DisSent/wikitext/'),
             'weakgrounded': (WeakGroundedTask, 'mscoco/weakgrounded/'),
             'grounded': (GroundedTask, 'mscoco/grounded/'),
             'ccg': (CCGTaggingTask, 'CCG/'),
             'nli-prob': (NLITypeProbingTask, 'NLI-Prob/'),
             'nli-alt': (NLITypeProbingAltTask, '/'),
             'nps': (NPSTask, 'nps/'), # NPS = Noun Phrases
             'vae': (VAETask, 'VAE'),
             'nli-alt': (NLITypeProbingAltTask, '/nfs/jsalt/exp/alexis-probing/results'),
             'recast-kg': (RecastKGTask, 'DNC/kg-relations'),
             'recast-lexicosyntax': (RecastLexicosynTask, 'DNC/lexicosyntactic_recasted'),
             'recast-winogender': (RecastWinogenderTask, 'DNC/manually-recast-winogender'),
             'recast-factuality': (RecastFactualityTask, 'DNC/recast_factuality_data'),
             'recast-ner': (RecastNERTask, 'DNC/recast_ner_data'),
             'recast-puns': (RecastPunTask, 'DNC/recast_puns_data'),
             'recast-sentiment': (RecastSentimentTask, 'DNC/recast_sentiment_data'),
             'recast-verbcorner': (RecastVerbcornerTask, 'DNC/recast_verbcorner_data'),
             'recast-verbnet': (RecastVerbnetTask, 'DNC/recast_verbnet_data'),
             'groundedsw': (GroundedSWTask, 'mscoco/grounded/'),
             }
# !!!!!!! NOTE: You should not be adding anything else manually to NAME2INFO.
# !!!!!!! Use the decorator @register_task instead.
# Add any tasks registered in tasks.py
NAME2INFO.update(tasks_module.REGISTRY)

SOS_TOK, EOS_TOK = "<SOS>", "<EOS>" # NOTE: these are not that same as AllenNLP SOS, EOS tokens
SPECIALS = [SOS_TOK, EOS_TOK] # NOTE: pad and unk tokens are created by AllenNLP vocabs by default
UNK_TOK = "@@UNKNOWN@@" # AllenNLP unk token

ALL_SPLITS = ['train', 'val', 'test']

def _get_serialized_record_path(task_name, split, preproc_dir):
    """Get the canonical path for a serialized task split."""
    serialized_record_path = os.path.join(preproc_dir,
                                          "{:s}__{:s}_data".format(task_name, split))
    return serialized_record_path


def _get_instance_generator(task_name, split, preproc_dir, fraction=None):
    """Get a lazy generator for the given task and split.

    Args:
        task_name: (string), task name
        split: (string), split name ('train', 'val', or 'test')
        preproc_dir: (string) path to preprocessing dir
        fraction: if set to a float between 0 and 1, load only the specified percentage
          of examples. Hashing is used to ensure that the same examples are loaded each
          epoch.

    Returns:
        serialize.RepeatableIterator yielding Instance objects
    """
    filename = _get_serialized_record_path(task_name, split, preproc_dir)
    assert os.path.isfile(filename), ("Record file '%s' not found!" % filename)
    return serialize.read_records(filename, repeatable=True, fraction=fraction)


def _indexed_instance_generator(instance_iter, vocab):
    """Yield indexed instances. Instances are modified in-place.

    TODO(iftenney): multiprocess the $%^& out of this.

    Args:
        instance_iter: iterable(Instance) of examples
        vocab: Vocabulary for use in indexing

    Yields:
        Instance with indexed fields.
    """
    for instance in instance_iter:
        instance.index_fields(vocab)
        # Strip token fields to save memory and disk.
        del_field_tokens(instance)
        yield instance


def del_field_tokens(instance):
    ''' Save memory by deleting the tokens that will no longer be used.
    Only works if Instances have fields 'input1' and 'input2'.
    All other fields will keep their tokens in memory.

    Args:
        instance: AllenNLP Instance. Modified in-place.
    '''
    if 'input1' in instance.fields:
        field = instance.fields['input1']
        del field.tokens
    if 'input2' in instance.fields:
        field = instance.fields['input2']
        del field.tokens


def _index_split(task, split, indexers, vocab, record_file):
    """Index instances and stream to disk.
    Args:
        task: Task instance
        split: (string), 'train', 'val', or 'test'
        indexers: dict of token indexers
        vocab: Vocabulary instance
        record_file: (string) file to write serialized Instances to
    """
    log_prefix = "\tTask '%s', split '%s'" % (task.name, split)
    log.info("%s: indexing from scratch", log_prefix)
    split_text = task.get_split_text(split)
    instance_iter = task.process_split(split_text, indexers)
    if hasattr(instance_iter, '__len__'):  # if non-lazy
        log.warn("%s: non-lazy Instance generation. You'll want to refactor "
                 "%s.process_split to return a lazy iterator.", log_prefix,
                 type(task).__name__)
        log.info("%s: %d examples to index", log_prefix, len(instance_iter))
        # Copy so that we don't store indexed data in memory.
        # TODO: remove this case and stream everything.
        instance_iter = utils.copy_iter(instance_iter)

    # Counter for lazy-loaded data, so we can log the # of elements.
    _instance_counter = 0
    def _counter_iter(elems):
        nonlocal _instance_counter
        for elem in elems:
            _instance_counter += 1
            yield elem
    instance_iter = _counter_iter(instance_iter)

    # Actually call generators and stream to disk.
    serialize.write_records(
        _indexed_instance_generator(instance_iter, vocab), record_file)
    log.info("%s: saved %d instances to %s",
             log_prefix, _instance_counter, record_file)

def _find_cached_file(exp_dir: str, global_exp_cache_dir: str,
                      relative_path: str, log_prefix: str="") -> bool:
    """Find a cached file.

    Look in local exp_dir first, then in global_exp_cache_dir. If found in the
    global dir, make a symlink in the local dir pointing to the global one.

    Args:
        exp_dir: (string) local experiment dir
        global_exp_cache_dir: (string) global experiment cache
        relative_path: (string) relative path to file, from exp_dir
        log_prefix: (string) prefix for logging info

    Returns:
        True if file was found in either location.
    """
    if log_prefix:
        log_prefix = log_prefix + ": "
    # Try in local preproc dir.
    local_file = os.path.join(exp_dir, relative_path)
    if os.path.isfile(local_file) or os.path.islink(local_file):
        log.info("%sFound preprocessed copy in %s", log_prefix, local_file)
        return True
    # Try in global preproc dir; if found, make a symlink.
    global_file = os.path.join(global_exp_cache_dir, relative_path)
    if os.path.exists(global_file):
        log.info("%sFound (global) preprocessed copy in %s", log_prefix, global_file)
        os.symlink(global_file, local_file)
        log.info("%sCreated symlink: %s -> %s", log_prefix, local_file, global_file)
        return True
    return False

def _build_embeddings(args, vocab, emb_file: str):
    ''' Build word embeddings from scratch (as opposed to loading them from a pickle),
    possibly using a fastText model or precomputed fastText / GloVe embeddings. '''
    log.info("\tBuilding embeddings from scratch")
    if args.fastText:
        word_embs, _ = get_fastText_model(vocab, args.d_word,
                                          model_file=args.fastText_model_file)
        log.info("\tUsing fastText; no pickling of embeddings.")
        return word_embs

    word_embs = get_embeddings(vocab, args.word_embs_file, args.d_word)
    pkl.dump(word_embs, open(emb_file, 'wb'))
    log.info("\tSaved embeddings to %s", emb_file)
    return word_embs

def _build_vocab(args, tasks, vocab_path: str):
    ''' Build vocabulary from scratch, reading data from tasks. '''
    # NOTE: task-specific target vocabulary should be counted in the task object
    # and provided via `task.all_labels()`. The namespace should be task-specific,
    # i.e. not something generic like "targets".
    log.info("\tBuilding vocab from scratch")
    max_v_sizes = {
        'word': args.max_word_v_size,
        'char': args.max_char_v_size,
    }
    word2freq, char2freq = get_words(tasks)
    vocab = get_vocab(word2freq, char2freq, max_v_sizes)
    for task in tasks:  # add custom label namespaces
        add_task_label_vocab(vocab, task)
    vocab.save_to_files(vocab_path)
    log.info("\tSaved vocab to %s", vocab_path)
    #  del word2freq, char2freq, target2freq

def build_tasks(args):
    '''Main logic for preparing tasks, doing so by
    1) creating / loading the tasks
    2) building / loading the vocabulary
    3) building / loading the word vectors
    4) indexing each task's data
    5) initializing lazy loaders (streaming iterators)
    '''

    # 1) create / load tasks
    tasks, train_task_names, eval_task_names = \
        get_tasks(parse_task_list_arg(args.train_tasks), parse_task_list_arg(args.eval_tasks), args.max_seq_len,
                  path=args.data_dir, scratch_path=args.exp_dir,
                  load_pkl=bool(not args.reload_tasks),
                  nli_prob_probe_path=args['nli-prob'].probe_path,
                  max_targ_v_size=args.max_targ_word_v_size)
    for task in tasks:
        task_classifier = config.get_task_attr(args, task.name, "use_classifier")
        setattr(task, "_classifier_name",
                task_classifier if task_classifier else task.name)

    # 2) build / load vocab and indexers
    vocab_path = os.path.join(args.exp_dir, 'vocab')
    indexers = {}
    if not args.word_embs == 'none':
        indexers["words"] = SingleIdTokenIndexer()
    if args.elmo:
        indexers["elmo"] = ELMoTokenCharactersIndexer("elmo")
    if args.char_embs:
        indexers["chars"] = TokenCharactersIndexer("chars")

    if args.reload_vocab or not os.path.exists(vocab_path):
        _build_vocab(args, tasks, vocab_path)

    # Always load vocab from file.
    vocab = Vocabulary.from_files(vocab_path)
    log.info("\tLoaded vocab from %s", vocab_path)

    for namespace, mapping in vocab._index_to_token.items():
        log.info("\tVocab namespace %s: size %d", namespace, len(mapping))
    log.info("\tFinished building vocab.")
    args.max_word_v_size = vocab.get_vocab_size('tokens')
    args.max_char_v_size = vocab.get_vocab_size('chars')

    # 3) build / load word vectors
    word_embs = None
    if args.word_embs != 'none':
        emb_file = os.path.join(args.exp_dir, 'embs.pkl')
        if args.reload_vocab or not os.path.exists(emb_file):
            word_embs = _build_embeddings(args, vocab, emb_file)
        else:  # load from file
            word_embs = pkl.load(open(emb_file, 'rb'))
    log.info("Trimmed word embeddings: %s", str(word_embs.size()))

    # 4) Index tasks using vocab (if preprocessed copy not available).
    preproc_dir = os.path.join(args.exp_dir, "preproc")
    utils.maybe_make_dir(preproc_dir)
    reindex_tasks = parse_task_list_arg(args.reindex_tasks)
    utils.assert_for_log(not (args.reload_indexing and not reindex_tasks),
                         "Flag reload_indexing was set, but no tasks are set to reindex (use -o \"args.reindex_tasks = \"task1,task2,...\"\")")
    for task in tasks:
        force_reindex = (args.reload_indexing and task.name in reindex_tasks)
        for split in ALL_SPLITS:
            log_prefix = "\tTask '%s', split '%s'" % (task.name, split)
            relative_path = _get_serialized_record_path(task.name, split, "preproc")
            cache_found = _find_cached_file(args.exp_dir, args.global_ro_exp_dir,
                                            relative_path, log_prefix=log_prefix)
            if force_reindex or not cache_found:
                # Re-index from scratch.
                record_file = _get_serialized_record_path(task.name, split,
                                                          preproc_dir)
                if os.path.exists(record_file) and os.path.islink(record_file):
                    os.remove(record_file)

                _index_split(task, split, indexers, vocab, record_file)

        # Delete in-memory data - we'll lazy-load from disk later.
        # TODO: delete task.{split}_data_text as well?
        task.train_data = None
        task.val_data = None
        task.test_data = None
        log.info("\tTask '%s': cleared in-memory data.", task.name)

    log.info("\tFinished indexing tasks")

    # 5) Initialize tasks with data iterators.
    assert not (args.training_data_fraction < 1 and args.eval_data_fraction < 1), \
        "training_data_fraction and eval_data_fraction could not be used at a same time (could not be < 1 together)"
    train_tasks = []
    eval_tasks = []
    for task in tasks:
        # Replace lists of instances with lazy generators from disk.
        task.val_data = _get_instance_generator(task.name, "val", preproc_dir)
        task.test_data = _get_instance_generator(task.name, "test", preproc_dir)
        # When using training_data_fraction, we need modified iterators for use
        # only on training datasets at pretraining time.
        if args.training_data_fraction < 1 and task.name in train_task_names:
            log.info("Creating trimmed pretraining-only version of " + task.name + " train.")
            task.train_data = _get_instance_generator(task.name, "train", preproc_dir,
                                                      fraction=args.training_data_fraction)
            train_tasks.append(task)
            if task.name in eval_task_names:
                # Rebuild the iterator so we see the full dataset in the eval training
                # phase. It will create a deepcopy of the task object
                # and therefore there could be two tasks with the same name (task.name).
                log.info("Creating un-trimmed eval training version of " + task.name + " train.")
                log.warn("When using un-trimmed eval training version of train split, "
                "it creates a deepcopy of task object which is inefficient.")
                task = copy.deepcopy(task)
                task.train_data = _get_instance_generator(
                    task.name, "train", preproc_dir, fraction=1.0)
                eval_tasks.append(task)

        # When using eval_data_fraction, we need modified iterators
        # only for training datasets at train_for_eval time.
        elif args.eval_data_fraction < 1 and task.name in eval_task_names:
            log.info("Creating trimmed train-for-eval-only version of " + task.name + " train.")
            task.train_data = _get_instance_generator(task.name, "train", preproc_dir,
                                                      fraction=args.eval_data_fraction)
            eval_tasks.append(task)
            if task.name in train_task_names:
                # Rebuild the iterator so we see the full dataset in the pretraining
                # phase. It will create a deepcopy of the task object
                # and therefore there could be two tasks with the same name (task.name).
                log.info("Creating un-trimmed pretraining version of " + task.name + " train.")
                log.warn("When using un-trimmed pretraining version of train split, "
                "it creates a deepcopy of task object which is inefficient.")
                task = copy.deepcopy(task)
                task.train_data = _get_instance_generator(
                    task.name, "train", preproc_dir, fraction=1.0)
                train_tasks.append(task)
        # When neither eval_data_fraction nor training_data_fraction is specified we use unmodified iterators.
        else:
            task.train_data = _get_instance_generator(task.name, "train", preproc_dir,
                                                      fraction=1.0)
            if task.name in train_task_names:
                train_tasks.append(task)
            if task.name in eval_task_names:
                eval_tasks.append(task)

        log.info("\tLazy-loading indexed data for task='%s' from %s",
                 task.name, preproc_dir)
    log.info("All tasks initialized with data iterators.")
    log.info('\t  Training on %s', ', '.join(train_task_names))
    log.info('\t  Evaluating on %s', ', '.join(eval_task_names))
    return train_tasks, eval_tasks, vocab, word_embs


def parse_task_list_arg(task_list):
    '''Parse task list argument into a list of task names.'''
    task_names = []
    for task_name in task_list.split(','):
        if task_name == 'glue':
            task_names.extend(ALL_GLUE_TASKS)
        elif task_name == 'edges-all':
            task_names.extend(ALL_EDGE_TASKS)
        elif task_name == 'none' or task_name == '':
            continue
        else:
            task_names.append(task_name)
    return task_names

def get_tasks(train_task_names, eval_task_names, max_seq_len, path=None,
              scratch_path=None, load_pkl=1, nli_prob_probe_path=None,
              max_targ_v_size=20000):
    ''' Actually build or load (from pickles) the tasks. '''
    # We don't want mnli-diagnostic in train_task_names
    train_task_names = [name for name in train_task_names if name not in {'mnli-diagnostic'}]
    ''' Load tasks '''
    task_names = sorted(set(train_task_names + eval_task_names))
    assert path is not None
    scratch_path = (scratch_path or path)
    log.info("Writing pre-preprocessed tasks to %s", scratch_path)

    tasks = []
    for name in task_names:
        assert name in NAME2INFO, "Task '{:s}' not found!".format(name)
        task_info = NAME2INFO[name]
        task_src_path = os.path.join(path, task_info[1])
        task_scratch_path = os.path.join(scratch_path, task_info[1])
        pkl_path = os.path.join(task_scratch_path, "%s_task.pkl" % name)
        if os.path.isfile(pkl_path) and load_pkl:
            task = pkl.load(open(pkl_path, 'rb'))
            log.info('\tLoaded existing task %s', name)
        else:
            log.info('\tCreating task %s from scratch', name)
            task_cls = task_info[0]
            kw = task_info[2] if len(task_info) > 2 else {}
            if name == 'nli-prob' or name=='nli-alt':  # this task takes additional kw
                # TODO: remove special case, replace with something general
                # to pass custom loader args to task.
                kw['probe_path'] = nli_prob_probe_path
            if name in ALL_TARG_VOC_TASKS:
                kw['max_targ_v_size'] = max_targ_v_size
            task = task_cls(task_src_path, max_seq_len, name=name, **kw)
            utils.maybe_make_dir(task_scratch_path)
            pkl.dump(task, open(pkl_path, 'wb'))
        #task.truncate(max_seq_len, SOS_TOK, EOS_TOK)

        # Count examples, store in example_counts.
        if not hasattr(task, 'example_counts'):
            task.count_examples()
        log.info("\tTask '%s': %s", task.name,
                 " ".join(("%s=%d" % kv for kv in
                           task.example_counts.items())))
        tasks.append(task)

    log.info("\tFinished loading tasks: %s.", ' '.join([task.name for task in tasks]))
    return tasks, train_task_names, eval_task_names


def get_words(tasks):
    '''
    Get all words for all tasks for all splits for all sentences
    Return dictionary mapping words to frequencies.
    '''
    word2freq, char2freq = defaultdict(int), defaultdict(int)

    def update_vocab_freqs(sentence):
        '''Update counts for words in the sentence'''
        for word in sentence:
            word2freq[word] += 1
            for char in list(word):
                char2freq[char] += 1
        return


    for task in tasks:
        log.info("\tCounting words for task: '%s'", task.name)
        if isinstance(task, MTTask):
            for src_sent, tgt_sent in task.get_sentences():
                update_vocab_freqs(src_sent)
        else:
            for sentence in task.get_sentences():
                update_vocab_freqs(sentence)

    # This branch is meant for tasks that have *English* target sentences
    # (or more generally, same language source and target sentences)
    # Tasks with different language source and target sentences should
    # count and return the vocab in a `task.all_labels()` method.
    for task in tasks:
        if hasattr(task, "target_sentences"):
            for sentence in task.target_sentences:
                update_target_vocab_freqs(sentence)

    log.info("\tFinished counting words")
    return word2freq, char2freq


def get_vocab(word2freq, char2freq, max_v_sizes):
    '''Build vocabulary by selecting the most frequent tokens'''
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

def add_task_label_vocab(vocab, task):
    '''Add custom task labels to a separate namespace.

    If task has a 'get_all_labels' method, call that to get a list of labels
    to populate the <task_name>_labels vocabulary namespace.

    This is the recommended way to implement multiclass models: in your task's
    process_split code, make instances that use LabelFields with the task label
    namespace, e.g.:
        label_namespace = "%s_labels" % self.name
        label = LabelField(label_string, label_namespace=label_namespace)
    This will cause them to be properly indexed by the Vocabulary.

    This can then be accessed when generating Instances, either via a custom
    Indexer or by invoking the namespace when creating a LabelField.
    '''
    if not hasattr(task, 'get_all_labels'):
        return
    utils.assert_for_log(hasattr(task, "_label_namespace"),
                         "Task %s is missing method `_label_namespace`!" % task.name)
    namespace = task._label_namespace
    if namespace is None:
        return
    log.info("\tTask '%s': adding vocab namespace '%s'", task.name, namespace)
    for label in task.get_all_labels():
        vocab.add_token_to_namespace(label, namespace)


def get_embeddings(vocab, vec_file, d_word) -> torch.FloatTensor:
    '''Get embeddings for the words in vocab from a file of precomputed vectors.
    Works for fastText and GloVe embedding files. '''
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
