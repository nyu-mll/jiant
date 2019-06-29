"""Preprocessing functions and pipeline

The pipeline is three steps
    1) create / load tasks, which includes
        a) load raw data
        b) tokenize raw data
    2) create / load all vocabularies (word, char, task-specific target vocabs)
        a) count tokens of a vocab
        b) take the N most frequent tokens
    3) index all the data using appropriate indexers
        We save indexed data to streamable Records to save memory.
"""
import _pickle as pkl  # :(
import copy
import io
import logging as log
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import (
    ELMoTokenCharactersIndexer,
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
)

from .tasks import ALL_COLA_NPI_TASKS, ALL_GLUE_TASKS, ALL_SUPERGLUE_TASKS, ALL_NLI_PROBING_TASKS
from .tasks import REGISTRY as TASKS_REGISTRY
from .utils import config, serialize, utils

# NOTE: these are not that same as AllenNLP SOS, EOS tokens
SOS_TOK, EOS_TOK = "<SOS>", "<EOS>"
# NOTE: pad and unk tokens are created by AllenNLP vocabs by default
SPECIALS = [SOS_TOK, EOS_TOK]
UNK_TOK = "@@UNKNOWN@@"  # AllenNLP unk token

ALL_SPLITS = ["train", "val", "test"]


def _get_serialized_record_path(task_name, split, preproc_dir):
    """Get the canonical path for a serialized task split."""
    serialized_record_path = os.path.join(preproc_dir, "{:s}__{:s}_data".format(task_name, split))
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
    assert os.path.isfile(filename), "Record file '%s' not found!" % filename
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
    """ Save memory by deleting the tokens that will no longer be used.
    Only works if Instances have fields 'input1' and 'input2'.
    All other fields will keep their tokens in memory.

    Args:
        instance: AllenNLP Instance. Modified in-place.
    """
    if "input1" in instance.fields:
        field = instance.fields["input1"]
        del field.tokens
    if "input2" in instance.fields:
        field = instance.fields["input2"]
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
    log_prefix = "\tTask %s (%s)" % (task.name, split)
    log.info("%s: Indexing from scratch.", log_prefix)
    split_text = task.get_split_text(split)
    instance_iter = task.process_split(split_text, indexers)
    if hasattr(instance_iter, "__len__"):  # if non-lazy
        log.warn(
            "%s: non-lazy Instance generation. You'll want to refactor "
            "%s.process_split to return a lazy iterator.",
            log_prefix,
            type(task).__name__,
        )
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
    serialize.write_records(_indexed_instance_generator(instance_iter, vocab), record_file)
    log.info("%s: Saved %d instances to %s", log_prefix, _instance_counter, record_file)


def _find_cached_file(
    exp_dir: str, global_exp_cache_dir: str, relative_path: str, log_prefix: str = ""
) -> bool:
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
    """ Build word embeddings from scratch (as opposed to loading them from a pickle),
    using precomputed fastText / GloVe embeddings. """

    # Load all the word embeddings based on vocabulary
    log.info("\tBuilding embeddings from scratch.")
    word_v_size, unk_idx = vocab.get_vocab_size("tokens"), vocab.get_token_index(vocab._oov_token)
    embeddings = np.random.randn(word_v_size, args.d_word)
    with io.open(
        args.word_embs_file, "r", encoding="utf-8", newline="\n", errors="ignore"
    ) as vec_fh:
        for line in vec_fh:
            word, vec = line.split(" ", 1)
            idx = vocab.get_token_index(word)
            if idx != unk_idx:
                embeddings[idx] = np.array(list(map(float, vec.split())))
    embeddings[vocab.get_token_index(vocab._padding_token)] = 0.0
    embeddings = torch.FloatTensor(embeddings)
    log.info("\tFinished loading embeddings")

    # Save/cache the word embeddings
    pkl.dump(embeddings, open(emb_file, "wb"))
    log.info("\tSaved embeddings to %s", emb_file)
    return embeddings


def _build_vocab(args, tasks, vocab_path: str):
    """ Build vocabulary from scratch, reading data from tasks. """
    # NOTE: task-specific target vocabulary should be counted in the task object
    # and provided via `task.all_labels()`. The namespace should be task-specific,
    # i.e. not something generic like "targets".
    log.info("\tBuilding vocab from scratch.")
    max_v_sizes = {"word": args.max_word_v_size, "char": args.max_char_v_size}
    word2freq, char2freq = get_words(tasks)
    vocab = get_vocab(word2freq, char2freq, max_v_sizes)
    for task in tasks:  # add custom label namespaces
        add_task_label_vocab(vocab, task)
    if args.force_include_wsj_vocabulary:
        # Add WSJ full vocabulary for PTB F1 parsing tasks.
        add_wsj_vocab(vocab, args.data_dir)
    if args.input_module == "gpt":
        # Add pre-computed BPE vocabulary for OpenAI transformer model.
        add_openai_bpe_vocab(vocab, "openai_bpe")
    if args.input_module.startswith("bert"):
        # Add pre-computed BPE vocabulary for BERT model.
        add_bert_wpm_vocab(vocab, args.input_module)

    vocab.save_to_files(vocab_path)
    log.info("\tSaved vocab to %s", vocab_path)
    #  del word2freq, char2freq, target2freq


def build_indexers(args):
    indexers = {}
    if not args.input_module.startswith("bert") and args.input_module not in ["elmo", "gpt"]:
        indexers["words"] = SingleIdTokenIndexer()
    if args.input_module == "elmo":
        indexers["elmo"] = ELMoTokenCharactersIndexer("elmo")
        assert args.tokenizer in {"", "MosesTokenizer"}
    if args.char_embs:
        indexers["chars"] = TokenCharactersIndexer("chars")
    if args.cove:
        assert args.tokenizer == "MosesTokenizer", (
            f"CoVe model expects Moses tokenization (MosesTokenizer);"
            " you are using args.tokenizer = {args.tokenizer}"
        )
    if args.input_module == "gpt":
        assert (
            not indexers
        ), "OpenAI transformer is not supported alongside other indexers due to tokenization."
        assert (
            args.tokenizer == "OpenAI.BPE"
        ), "OpenAI transformer uses custom BPE tokenization. Set tokenizer=OpenAI.BPE."
        indexers["openai_bpe_pretokenized"] = SingleIdTokenIndexer("openai_bpe")
    if args.input_module.startswith("bert"):
        assert not indexers, "BERT is not supported alongside other indexers due to tokenization."
        assert args.tokenizer == args.input_module, (
            "BERT models use custom WPM tokenization for "
            "each model, so tokenizer must match the "
            "specified BERT model."
        )
        indexers["bert_wpm_pretokenized"] = SingleIdTokenIndexer(args.input_module)
    return indexers


def build_tasks(args):
    """Main logic for preparing tasks, doing so by
    1) creating / loading the tasks
    2) building / loading the vocabulary
    3) building / loading the word vectors
    4) indexing each task's data
    5) initializing lazy loaders (streaming iterators)
    """

    # 1) create / load tasks
    tasks, pretrain_task_names, target_task_names = get_tasks(args)
    for task in tasks:
        task_classifier = config.get_task_attr(args, task.name, "use_classifier")
        setattr(task, "_classifier_name", task_classifier if task_classifier else task.name)

    tokenizer_names = {task.name: task.tokenizer_name for task in tasks}
    assert len(set(tokenizer_names.values())) == 1, (
        f"Error: mixing tasks with different tokenizers!" " Tokenizations: {tokenizer_names:s}"
    )

    # 2) build / load vocab and indexers
    indexers = build_indexers(args)

    vocab_path = os.path.join(args.exp_dir, "vocab")
    if args.reload_vocab or not os.path.exists(vocab_path):
        _build_vocab(args, tasks, vocab_path)

    # Always load vocab from file.
    vocab = Vocabulary.from_files(vocab_path)
    log.info("\tLoaded vocab from %s", vocab_path)

    for namespace, mapping in vocab._index_to_token.items():
        log.info("\tVocab namespace %s: size %d", namespace, len(mapping))
    log.info("\tFinished building vocab.")
    args.max_word_v_size = vocab.get_vocab_size("tokens")
    args.max_char_v_size = vocab.get_vocab_size("chars")

    # 3) build / load word vectors
    word_embs = None
    if args.input_module not in ["elmo", "gpt", "scratch"] and not args.input_module.startswith(
        "bert"
    ):
        emb_file = os.path.join(args.exp_dir, "embs.pkl")
        if args.reload_vocab or not os.path.exists(emb_file):
            word_embs = _build_embeddings(args, vocab, emb_file)
        else:  # load from file
            word_embs = pkl.load(open(emb_file, "rb"))
        log.info("Trimmed word embeddings: %s", str(word_embs.size()))

    # 4) Index tasks using vocab (if preprocessed copy not available).
    preproc_dir = os.path.join(args.exp_dir, "preproc")
    utils.maybe_make_dir(preproc_dir)
    reindex_tasks = parse_task_list_arg(args.reindex_tasks)
    utils.assert_for_log(
        not (args.reload_indexing and not reindex_tasks),
        'Flag reload_indexing was set, but no tasks are set to reindex (use -o "args.reindex_tasks'
        ' = "task1,task2,..."")',
    )
    for task in tasks:
        force_reindex = args.reload_indexing and task.name in reindex_tasks
        for split in ALL_SPLITS:
            log_prefix = "\tTask '%s', split '%s'" % (task.name, split)
            relative_path = _get_serialized_record_path(task.name, split, "preproc")
            cache_found = _find_cached_file(
                args.exp_dir, args.global_ro_exp_dir, relative_path, log_prefix=log_prefix
            )
            if force_reindex or not cache_found:
                # Re-index from scratch.
                record_file = _get_serialized_record_path(task.name, split, preproc_dir)
                if os.path.exists(record_file) and os.path.islink(record_file):
                    os.remove(record_file)

                _index_split(task, split, indexers, vocab, record_file)

        # Delete in-memory data - we'll lazy-load from disk later.
        # TODO: delete task.{split}_data_text as well?
        task.train_data = None
        task.val_data = None
        task.test_data = None

    log.info("\tFinished indexing tasks")

    # 5) Initialize tasks with data iterators.
    pretrain_tasks = []
    target_tasks = []
    for task in tasks:
        # Replace lists of instances with lazy generators from disk.
        task.val_data = _get_instance_generator(task.name, "val", preproc_dir)
        task.test_data = _get_instance_generator(task.name, "test", preproc_dir)
        # When using pretrain_data_fraction, we need modified iterators for use
        # only on training datasets at pretraining time.
        if task.name in pretrain_task_names:
            log.info("\tCreating trimmed pretraining-only version of " + task.name + " train.")
            task.train_data = _get_instance_generator(
                task.name, "train", preproc_dir, fraction=args.pretrain_data_fraction
            )
            pretrain_tasks.append(task)
        # When using target_train_data_fraction, we need modified iterators
        # only for training datasets at do_target_task_training time.
        if task.name in target_task_names:
            log.info("\tCreating trimmed target-only version of " + task.name + " train.")
            task.train_data = _get_instance_generator(
                task.name, "train", preproc_dir, fraction=args.target_train_data_fraction
            )
            target_tasks.append(task)

    log.info("\t  Training on %s", ", ".join(pretrain_task_names))
    log.info("\t  Evaluating on %s", ", ".join(target_task_names))
    return pretrain_tasks, target_tasks, vocab, word_embs


def parse_task_list_arg(task_list):
    """Parse task list argument into a list of task names."""
    task_names = []
    for task_name in task_list.split(","):
        if task_name == "glue":
            task_names.extend(ALL_GLUE_TASKS)
        elif task_name == "superglue":
            task_names.extend(ALL_SUPERGLUE_TASKS)
        elif task_name == "none" or task_name == "":
            continue
        else:
            task_names.append(task_name)
    return task_names


def _get_task(name, args, data_path, scratch_path):
    """ Build or load a single task. """
    assert name in TASKS_REGISTRY, f"Task '{name:s}' not found!"
    task_cls, rel_path, task_kw = TASKS_REGISTRY[name]
    pkl_path = os.path.join(scratch_path, "tasks", f"{name:s}.{args.tokenizer:s}.pkl")
    # TODO: refactor to always read from disk, even if task is constructed
    # here. This should avoid subtle bugs from deserialization issues.
    if os.path.isfile(pkl_path) and not args.reload_tasks:
        task = pkl.load(open(pkl_path, "rb"))
        log.info("\tLoaded existing task %s", name)
    else:
        log.info("\tCreating task %s from scratch.", name)
        # These tasks take an additional kwarg.
        if name == "nli-prob" or name == "nli-alt":
            # TODO: remove special case, replace with something general
            # to pass custom loader args to task.
            task_kw["probe_path"] = args["nli-prob"].probe_path
        task_src_path = os.path.join(data_path, rel_path)
        task = task_cls(
            task_src_path,
            max_seq_len=args.max_seq_len,
            name=name,
            tokenizer_name=args.tokenizer,
            **task_kw,
        )
        task.load_data()
        utils.maybe_make_dir(os.path.dirname(pkl_path))
        pkl.dump(task, open(pkl_path, "wb"))

    return task


def get_task_without_loading_data(task_name, args):
    """ Build a task without loading data """
    task_cls, rel_path, task_kw = TASKS_REGISTRY[task_name]
    task = task_cls(
        path=None,
        max_seq_len=args.max_seq_len,
        name=task_name,
        tokenizer_name=args.tokenizer,
        **task_kw,
    )
    return task


def get_tasks(args):
    """ Actually build or load (from pickles) the tasks. """
    data_path = args.data_dir
    scratch_path = args.exp_dir

    pretrain_task_names = parse_task_list_arg(args.pretrain_tasks)
    target_task_names = parse_task_list_arg(args.target_tasks)
    # TODO: We don't want diagnostic tasks in train_task_names
    # but want to support glue/superglue task macros.
    # A solution that doesn't rely on enumerating names would be nice.
    pretrain_task_names = [
        name
        for name in pretrain_task_names
        if name not in {"glue-diagnostic", "superglue-diagnostic"}
    ]

    task_names = sorted(set(pretrain_task_names + target_task_names))
    assert data_path is not None
    scratch_path = scratch_path or data_path
    log.info("Writing pre-preprocessed tasks to %s", scratch_path)

    tasks = []
    for name in task_names:
        task = _get_task(name, args, data_path=data_path, scratch_path=scratch_path)
        tasks.append(task)

        # Count examples, store in example_counts.
        if task.example_counts is None:
            task.count_examples()
        log.info(
            "\tTask '%s': %s",
            task.name,
            " ".join(("|%s|=%d" % kv for kv in task.example_counts.items())),
        )

    log.info("\tFinished loading tasks: %s.", " ".join([task.name for task in tasks]))
    return tasks, pretrain_task_names, target_task_names


def get_words(tasks):
    """
    Get all words for all tasks for all splits for all sentences
    Return dictionary mapping words to frequencies.
    """
    word2freq, char2freq = defaultdict(int), defaultdict(int)

    def update_vocab_freqs(sentence):
        """Update counts for words in the sentence"""
        for word in sentence:
            word2freq[word] += 1
            for char in list(word):
                char2freq[char] += 1
        return

    for task in tasks:
        log.info("\tCounting words for task %s.", task.name)
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

    return word2freq, char2freq


def get_vocab(word2freq, char2freq, max_v_sizes):
    """Build vocabulary by selecting the most frequent tokens"""
    vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes)
    for special in SPECIALS:
        vocab.add_token_to_namespace(special, "tokens")

    words_by_freq = [(word, freq) for word, freq in word2freq.items()]
    words_by_freq.sort(key=lambda x: x[1], reverse=True)
    for word, _ in words_by_freq[: max_v_sizes["word"]]:
        vocab.add_token_to_namespace(word, "tokens")

    chars_by_freq = [(char, freq) for char, freq in char2freq.items()]
    chars_by_freq.sort(key=lambda x: x[1], reverse=True)
    for char, _ in chars_by_freq[: max_v_sizes["char"]]:
        vocab.add_token_to_namespace(char, "chars")

    return vocab


def add_task_label_vocab(vocab, task):
    """Add custom task labels to a separate namespace.

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
    """
    if not hasattr(task, "get_all_labels"):
        return
    utils.assert_for_log(
        hasattr(task, "_label_namespace"),
        "Task %s is missing method `_label_namespace`!" % task.name,
    )
    namespace = task._label_namespace
    if namespace is None:
        return
    log.info("\tTask '%s': adding vocab namespace '%s'", task.name, namespace)
    for label in task.get_all_labels():
        vocab.add_token_to_namespace(label, namespace)


def add_bert_wpm_vocab(vocab, bert_model_name):
    """Add BERT WPM vocabulary for use with pre-tokenized data.

    BertTokenizer has a convert_tokens_to_ids method, but this doesn't do
    anything special so we can just use the standard indexers.
    """
    from pytorch_pretrained_bert import BertTokenizer

    do_lower_case = bert_model_name.endswith("uncased")
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=do_lower_case)
    ordered_vocab = tokenizer.convert_ids_to_tokens(range(len(tokenizer.vocab)))
    log.info("BERT WPM vocab (model=%s): %d tokens", bert_model_name, len(ordered_vocab))
    for word in ordered_vocab:
        vocab.add_token_to_namespace(word, bert_model_name)


def add_openai_bpe_vocab(vocab, namespace="openai_bpe"):
    """Add OpenAI BPE vocabulary for use with pre-tokenized data."""
    from .openai_transformer_lm import utils as openai_utils

    id_to_wordpiece = openai_utils.reverse_encoder_dict
    for i in range(len(id_to_wordpiece)):
        vocab.add_token_to_namespace(id_to_wordpiece[i], namespace)
    # Add SOS and EOS tokens to *end* of namespace, since this is where the
    # OpenAI model expects special tokens.
    vocab.add_token_to_namespace(utils.SOS_TOK, namespace)
    vocab.add_token_to_namespace(utils.EOS_TOK, namespace)


def add_wsj_vocab(vocab, data_dir, namespace="tokens"):
    """Add WSJ vocabulary for PTB parsing models."""
    wsj_vocab_path = os.path.join(data_dir, "WSJ/tokens.txt")
    # To create the tokens.txt file: Run only WSJ LM baseline on jiant, and
    # duplicate the vocab file generated.
    assert os.path.exists(wsj_vocab_path), "WSJ vocab file doesn't exist."
    wsj_tokens = open(wsj_vocab_path)
    for line in wsj_tokens.readlines():
        vocab.add_token_to_namespace(line.strip(), namespace)
    log.info("\tAdded WSJ vocabulary from %s", wsj_tokens)
