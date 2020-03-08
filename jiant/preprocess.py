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
from typing import List, Dict, Union, Any

import numpy as np
import torch
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import (
    ELMoTokenCharactersIndexer,
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
)

from jiant.huggingface_transformers_interface import (
    input_module_uses_transformers,
    input_module_tokenizer_name,
)
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    AlbertTokenizer,
    XLNetTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    TransfoXLTokenizer,
    XLMTokenizer,
)

from jiant.tasks import (
    ALL_DIAGNOSTICS,
    ALL_COLA_NPI_TASKS,
    ALL_GLUE_TASKS,
    ALL_SUPERGLUE_TASKS,
    ALL_NLI_PROBING_TASKS,
    ALL_SEQ2SEQ_TASKS,
)
from jiant.tasks.lm import MLMTask
from jiant.tasks import REGISTRY as TASKS_REGISTRY
from jiant.tasks.seq2seq import Seq2SeqTask
from jiant.tasks.tasks import SequenceGenerationTask, Task
from jiant.utils import config, serialize, utils, options
from jiant.utils.options import parse_task_list_arg

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


def _index_split(task, split, indexers, vocab, record_file, model_preprocessing_interface):
    """Index instances and stream to disk.
    Args:
        task: Task instance
        split: (string), 'train', 'val', or 'test'
        indexers: dict of token indexers
        vocab: Vocabulary instance
        record_file: (string) file to write serialized Instances to
        model_preprocessing_interface: packed information from model that effects the task data,
            including whether to concatenate sentence pair, and how to mark the sentence boundry
    """
    log_prefix = "\tTask %s (%s)" % (task.name, split)
    log.info("%s: Indexing from scratch.", log_prefix)
    split_text = task.get_split_text(split)
    instance_iter = task.process_split(split_text, indexers, model_preprocessing_interface)
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

    if args.word_embs_file:
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


def _build_vocab(args: config.Params, tasks: List[Task], vocab_path: str):
    """Build vocabulary from scratch

    Read data from all tasks into namespaces, optionally add special vocab items, and save
    vocabulary file.

    Note
    ----
    task-specific target vocabulary should be counted in the task object
    and provided via `task.all_labels()`. The namespace should be task-specific,
    i.e. not something generic like "targets".

    Parameters
    ----------
    args : config.Params
        config map
    tasks : List[Task]
        list of Task from which to build vocab
    vocab_path : str
        vocab file save path

    """
    log.info("\tBuilding vocab from scratch.")
    max_v_sizes = {"word": args.max_word_v_size, "char": args.max_char_v_size}
    word2freq, char2freq = get_words(tasks)
    vocab = get_vocab(word2freq, char2freq, max_v_sizes)

    if args.force_include_wsj_vocabulary:
        # Add WSJ full vocabulary for PTB F1 parsing tasks.
        add_wsj_vocab(vocab, args.data_dir)
    if input_module_uses_transformers(args.input_module):
        # Add pre-computed vocabulary of corresponding tokenizer for transformers models.
        add_transformers_vocab(vocab, args.tokenizer)
    for task in tasks:  # add custom label namespaces
        # TODO: surface more docs for add_task_label_vocab:
        add_task_label_vocab(vocab, task)

    vocab.save_to_files(vocab_path)
    log.info("\tSaved vocab to %s", vocab_path)
    #  del word2freq, char2freq, target2freq


def build_indexers(args):
    indexers = {}
    if args.input_module in ["scratch", "glove", "fastText"]:
        indexers["words"] = SingleIdTokenIndexer()
    elif args.input_module in ["elmo", "elmo-chars-only"]:
        indexers["elmo"] = ELMoTokenCharactersIndexer("elmo")
        assert args.tokenizer in {"", "MosesTokenizer"}

    if args.char_embs:
        indexers["chars"] = TokenCharactersIndexer("chars")
    if args.cove:
        assert args.tokenizer == "MosesTokenizer", (
            f"CoVe model expects Moses tokenization (MosesTokenizer);"
            " you are using args.tokenizer = {args.tokenizer}"
        )

    if input_module_uses_transformers(args.input_module):
        assert (
            not indexers
        ), "transformers modules like BERT/XLNet are not supported alongside other "
        "indexers due to tokenization."
        assert args.tokenizer == args.input_module, (
            "transformers models use custom tokenization for each model, so tokenizer "
            "must match the specified model."
        )
        tokenizer_name = input_module_tokenizer_name(args.input_module)
        indexers[tokenizer_name] = SingleIdTokenIndexer(tokenizer_name)
    return indexers


def build_tasks(
    args: config.Params, cuda_device: Any
) -> (List[Task], List[Task], Vocabulary, Union[np.ndarray, float]):
    """Main logic for preparing tasks:

    1. create or load the tasks
    2. configure classifiers for tasks
    3. set up indexers
    4. build and save vocab to disk
    5. load vocab from disk
    6. if specified, load word embeddings
    7. set up ModelPreprocessingInterface (MPI) to handle model-specific preprocessing
    8. index tasks using vocab and task-specific MPI, save to disk.
    9. return: task data lazy-loaders in phase-specific lists w/ vocab, and word embeddings

    Parameters
    ----------
    args : Params
        config map

    Returns
    -------
    List[Task]
        list of pretrain Tasks.
    List[Task]
        list of target Tasks.
    allennlp.data.Vocabulary
        vocabulary from task data.
    Union[np.ndarray, float]
        Word embeddings.

    """
    # 1) create / load tasks
    tasks, pretrain_task_names, target_task_names = get_tasks(args, cuda_device)
    for task in tasks:
        task_classifier = config.get_task_attr(args, task.name, "use_classifier")
        setattr(task, "_classifier_name", task_classifier if task_classifier else task.name)

    tokenizer_names = {task.name: task.tokenizer_name for task in tasks}
    assert not len(set(tokenizer_names.values())) > 1, (
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
    if args.input_module in ["glove", "fastText"]:
        emb_file = os.path.join(args.exp_dir, "embs.pkl")
        if args.reload_vocab or not os.path.exists(emb_file):
            word_embs = _build_embeddings(args, vocab, emb_file)
        else:  # load from file
            word_embs = pkl.load(open(emb_file, "rb"))
        log.info("Trimmed word embeddings: %s", str(word_embs.size()))

    # 4) Set up model_preprocessing_interface
    model_preprocessing_interface = ModelPreprocessingInterface(args)

    # 5) Index tasks using vocab (if preprocessed copy not available).
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

                _index_split(
                    task, split, indexers, vocab, record_file, model_preprocessing_interface
                )

        # Delete in-memory data - we'll lazy-load from disk later.
        # TODO: delete task.{split}_data_text?

    log.info("\tFinished indexing tasks")

    # 6) Initialize tasks with data iterators.
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


def _get_task(name: str, args: config.Params, data_path: str, scratch_path: str) -> Task:
    """Get task object from disk if available. Else construct, prepare and save a new task object.

    Parameters
    ----------
    name : str
        task name to load.
    args : config.Params
        param handler object.
    data_path : str
        base data directory.
    scratch_path : str
        where to save Task objects.

    Returns
    -------
    Task
        loaded task object.

    """
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
        if name in ALL_SEQ2SEQ_TASKS:
            task_kw["max_targ_v_size"] = args.max_targ_word_v_size
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


def get_tasks(args: config.Params, cuda_device: Any) -> (List[Task], List[str], List[str]):
    """Get and save tasks:

    1. Set up task storage file paths
    2. Parse config for task names
    3. Load (or build and save) task objects
    4. Call counting methods on task objects
    5. Log example-count stats for tasks.

    Parameters
    ----------
    args : config.Params
        config map.

    Returns
    -------
    List[Task]
        list of all loaded Tasks.
    List[str]
        pretrain task names.
    List[str]
        target task names.

    """
    data_path = args.data_dir
    scratch_path = args.exp_dir

    pretrain_task_names = parse_task_list_arg(args.pretrain_tasks)
    target_task_names = parse_task_list_arg(args.target_tasks)
    # TODO: We don't want diagnostic tasks in train_task_names
    # but want to support glue/superglue task macros.
    pretrain_task_names = list(filter(lambda x: x not in ALL_DIAGNOSTICS, pretrain_task_names))

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


def get_words(tasks: List[Task]) -> (Dict[str, int], Dict[str, int]):
    """Get all words for all tasks for all splits for all sentences across all tasks.

    Parameters
    ----------
    tasks : List[Task]
        List of tasks to process.

    Returns
    -------
    Dict[str, int]
        Dictionary storing word frequencies across all tasks.
    Dict[str, int]
        Dictionary storing char frequencies across all tasks.

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
        log.info("\tCounting units for task %s.", task.name)
        if isinstance(task, Seq2SeqTask):
            for src_sent, tgt_sent in task.get_sentences():
                update_vocab_freqs(src_sent)
        else:
            for sentence in task.get_sentences():
                update_vocab_freqs(sentence)

    return word2freq, char2freq


def get_vocab(
    word2freq: Dict[str, int], char2freq: Dict[str, int], max_v_sizes: Dict[str, int]
) -> Vocabulary:
    """Build vocabulary by selecting the most frequent tokens

    Parameters
    ----------
    word2freq : Dict[str, int]
        Dict mapping words to frequencies.
    char2freq : Dict[str, int]
        Dict mapping chars to frequencies.
    max_v_sizes : dict[str: int]
        Dict used to set max vocab size for each token namespace.

    Returns
    -------
    allennlp.data.Vocabulary
        vocab containing word and char namespaces.

    """
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

    if isinstance(task, SequenceGenerationTask):
        for special in SPECIALS:
            vocab.add_token_to_namespace(special, namespace)

    for label in task.get_all_labels():
        vocab.add_token_to_namespace(label, namespace)


def add_transformers_vocab(vocab, tokenizer_name):
    """Add vocabulary from tokenizers in transformers for use with pre-tokenized data.

    These tokenizers have a convert_tokens_to_ids method, but this doesn't do
    anything special, so we can just use the standard indexers.
    """
    do_lower_case = "uncased" in tokenizer_name

    if tokenizer_name.startswith("bert-"):
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=do_lower_case)
    elif tokenizer_name.startswith("roberta-"):
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("albert-"):
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("xlnet-"):
        tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name, do_lower_case=do_lower_case)
    elif tokenizer_name.startswith("openai-gpt"):
        tokenizer = OpenAIGPTTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("transfo-xl-"):
        tokenizer = TransfoXLTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name.startswith("xlm-"):
        tokenizer = XLMTokenizer.from_pretrained(tokenizer_name)

    if (
        tokenizer_name.startswith("openai-gpt")
        or tokenizer_name.startswith("gpt2")
        or tokenizer_name.startswith("transo-xl-")
    ):
        tokenizer.add_special_tokens(
            {"bos_token": "<start>", "sep_token": "<delim>", "cls_token": "<extract>"}
        )
    # TODO: this is another place can be simplified by "model-before-preprocess" reorganization
    # we can pass tokenizer created in model here, see issue <TBD>

    vocab_size = len(tokenizer)
    # do not use tokenizer.vocab_size, it does not include newly added token

    ordered_vocab = tokenizer.convert_ids_to_tokens(range(vocab_size))
    log.info("Added transformers vocab (%s): %d tokens", tokenizer_name, len(ordered_vocab))
    for word in ordered_vocab:
        vocab.add_token_to_namespace(word, input_module_tokenizer_name(tokenizer_name))


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


class ModelPreprocessingInterface(object):
    """ This class holds parts of preprocessing that is model-specific
    members:

    model_flags: Dict[str, bool], model-specific flags that may be used in task preprocessing
    boundary_token_fn: (list[str], list[str] (optional) -> list[str]):
        A function that appliese the appropriate EOS/SOS/SEP/CLS tokens to token sequence or
        token sequence pair for most tasks.
    lm_boundary_token_fn: (list[str] -> list[str]):
        A function that appliese the appropriate EOS/SOS/SEP/CLS tokens to a token sequence for
        language modeling tasks.

    """

    def __init__(self, args):
        boundary_token_fn = None
        lm_boundary_token_fn = None

        if args.input_module.startswith("bert-"):
            from jiant.huggingface_transformers_interface.modules import BertEmbedderModule

            boundary_token_fn = BertEmbedderModule.apply_boundary_tokens
        elif args.input_module.startswith("roberta-"):
            from jiant.huggingface_transformers_interface.modules import RobertaEmbedderModule

            boundary_token_fn = RobertaEmbedderModule.apply_boundary_tokens
        elif args.input_module.startswith("albert-"):
            from jiant.huggingface_transformers_interface.modules import AlbertEmbedderModule

            boundary_token_fn = AlbertEmbedderModule.apply_boundary_tokens
        elif args.input_module.startswith("xlnet-"):
            from jiant.huggingface_transformers_interface.modules import XLNetEmbedderModule

            boundary_token_fn = XLNetEmbedderModule.apply_boundary_tokens
        elif args.input_module.startswith("openai-gpt"):
            from jiant.huggingface_transformers_interface.modules import OpenAIGPTEmbedderModule

            boundary_token_fn = OpenAIGPTEmbedderModule.apply_boundary_tokens
            lm_boundary_token_fn = OpenAIGPTEmbedderModule.apply_lm_boundary_tokens
        elif args.input_module.startswith("gpt2"):
            from jiant.huggingface_transformers_interface.modules import GPT2EmbedderModule

            boundary_token_fn = GPT2EmbedderModule.apply_boundary_tokens
            lm_boundary_token_fn = GPT2EmbedderModule.apply_lm_boundary_tokens
        elif args.input_module.startswith("transfo-xl-"):
            from jiant.huggingface_transformers_interface.modules import TransfoXLEmbedderModule

            boundary_token_fn = TransfoXLEmbedderModule.apply_boundary_tokens
            lm_boundary_token_fn = TransfoXLEmbedderModule.apply_lm_boundary_tokens
        elif args.input_module.startswith("xlm-"):
            from jiant.huggingface_transformers_interface.modules import XLMEmbedderModule

            boundary_token_fn = XLMEmbedderModule.apply_boundary_tokens
        else:
            boundary_token_fn = utils.apply_standard_boundary_tokens

        self.boundary_token_fn = boundary_token_fn
        if lm_boundary_token_fn is not None:
            self.lm_boundary_token_fn = lm_boundary_token_fn
        else:
            self.lm_boundary_token_fn = boundary_token_fn

        from jiant.models import input_module_uses_pair_embedding, input_module_uses_mirrored_pair

        self.model_flags = {}
        self.model_flags["uses_pair_embedding"] = input_module_uses_pair_embedding(
            args.input_module
        )
        self.model_flags["uses_mirrored_pair"] = input_module_uses_mirrored_pair(args.input_module)
