'''Core model and functions for building it.'''
import os
import sys
import math
import copy
import json
import logging as log

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error

from allennlp.common import Params
from allennlp.modules import Elmo, Seq2SeqEncoder, SimilarityFunction, TimeDistributed
from allennlp.nn import util
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.training.metrics import Average

from .allennlp_mods.elmo_text_field_embedder import ElmoTextFieldEmbedder, ElmoTokenEmbedderWrapper
from .utils.utils import assert_for_log, get_batch_utilization, \
    get_batch_size, get_elmo_mixing_weights, maybe_make_dir
from .utils import config

from .preprocess import parse_task_list_arg, get_tasks

from .tasks.tasks import CCGTaggingTask, ClassificationTask, CoLATask, CoLAAnalysisTask, \
    GroundedSWTask, GroundedTask, MultiNLIDiagnosticTask, PairClassificationTask, \
    PairOrdinalRegressionTask, PairRegressionTask, RankingTask, \
    RegressionTask, SequenceGenerationTask, SingleClassificationTask, SSTTask, STSBTask, \
    TaggingTask, WeakGroundedTask, JOCITask
from .tasks.lm import LanguageModelingTask
from .tasks.mt import MTTask, RedditSeq2SeqTask, Wiki103Seq2SeqTask
from .tasks.edge_probing import EdgeProbingTask

from .modules.modules import SentenceEncoder, BoWSentEncoder, \
    AttnPairEncoder, MaskedStackedSelfAttentionEncoder, \
    BiLMEncoder, ElmoCharacterEncoder, Classifier, Pooler, \
    SingleClassifier, PairClassifier, CNNEncoder, \
    NullPhraseLayer
from .modules.edge_probing import EdgeClassifierModule
from .modules.seq2seq_decoder import Seq2SeqDecoder


# Elmo stuff
# Look in $ELMO_SRC_DIR (e.g. /usr/share/jsalt/elmo) or download from web
ELMO_OPT_NAME = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHTS_NAME = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
ELMO_SRC_DIR = (os.getenv("ELMO_SRC_DIR") or
                "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/")
ELMO_OPT_PATH = os.path.join(ELMO_SRC_DIR, ELMO_OPT_NAME)
ELMO_WEIGHTS_PATH = os.path.join(ELMO_SRC_DIR, ELMO_WEIGHTS_NAME)


def build_sent_encoder(args, vocab, d_emb, tasks, embedder, cove_layer):
    # Build single sentence encoder: the main component of interest
    # Need special handling for language modeling
    # Note: sent_enc is expected to apply dropout to its input _and_ output if
    # needed.
    tfm_params = Params({'input_dim': d_emb, 'hidden_dim': args.d_hid,
                         'projection_dim': args.d_tproj,
                         'feedforward_hidden_dim': args.d_ff,
                         'num_layers': args.n_layers_enc,
                         'num_attention_heads': args.n_heads})
    rnn_params = Params({'input_size': d_emb, 'bidirectional': True,
                         'hidden_size': args.d_hid, 'num_layers': args.n_layers_enc})
    # Make sentence encoder
    if any(isinstance(task, LanguageModelingTask) for task in tasks) or \
            args.sent_enc == 'bilm':
        assert_for_log(
            args.sent_enc in [
                'rnn',
                'bilm'],
            "Only RNNLM supported!")
        if args.elmo:
            assert_for_log(
                args.elmo_chars_only,
                "LM with full ELMo not supported")
        bilm = BiLMEncoder(d_emb, args.d_hid, args.d_hid, args.n_layers_enc)
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       bilm, skip_embs=args.skip_embs,
                                       dropout=args.dropout,
                                       sep_embs_for_skip=args.sep_embs_for_skip,
                                       cove_layer=cove_layer)
        d_sent = 2 * args.d_hid
        log.info("Using BiLM architecture for shared encoder!")
    elif args.sent_enc == 'bow':
        sent_encoder = BoWSentEncoder(vocab, embedder)
        log.info("Using BoW architecture for shared encoder!")
        assert_for_log(
            not args.skip_embs,
            "Skip connection not currently supported with `bow` encoder.")
        d_sent = d_emb
    elif args.sent_enc == 'rnn':
        sent_rnn = s2s_e.by_name('lstm').from_params(copy.deepcopy(rnn_params))
        sent_encoder = SentenceEncoder(
            vocab,
            embedder,
            args.n_layers_highway,
            sent_rnn,
            skip_embs=args.skip_embs,
            dropout=args.dropout,
            sep_embs_for_skip=args.sep_embs_for_skip,
            cove_layer=cove_layer)
        d_sent = 2 * args.d_hid
        log.info("Using BiLSTM architecture for shared encoder!")
    elif args.sent_enc == 'transformer':
        transformer = StackedSelfAttentionEncoder.from_params(
            copy.deepcopy(tfm_params))
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       transformer, dropout=args.dropout,
                                       skip_embs=args.skip_embs,
                                       cove_layer=cove_layer,
                                       sep_embs_for_skip=args.sep_embs_for_skip)
        log.info("Using Transformer architecture for shared encoder!")
    elif args.sent_enc == 'null':
        # Expose word representation layer (GloVe, ELMo, etc.) directly.
        assert_for_log(args.skip_embs, f"skip_embs must be set for "
                       "'{args.sent_enc}' encoder")
        phrase_layer = NullPhraseLayer(rnn_params['input_size'])
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       phrase_layer, skip_embs=args.skip_embs,
                                       dropout=args.dropout,
                                       sep_embs_for_skip=args.sep_embs_for_skip,
                                       cove_layer=cove_layer)
        d_sent = 0  # skip connection added below
        log.info("No shared encoder (just using word embeddings)!")
    else:
        assert_for_log(False, "No valid sentence encoder specified.")
    return sent_encoder, d_sent


def build_model(args, vocab, pretrained_embs, tasks):
    '''
    Build model according to args
    Returns: model which has attributes set in it with the attrbutes.
    '''

    # Build embeddings.
    if args.openai_transformer:
        # Note: incompatible with other embedders, but logic in preprocess.py
        # should prevent these from being enabled anyway.
        from .openai_transformer_lm.utils import OpenAIEmbedderModule
        log.info("Using OpenAI transformer model; skipping other embedders.")
        cove_layer = None
        # Here, this uses openAIEmbedder.
        embedder = OpenAIEmbedderModule(args)
        d_emb = embedder.get_output_dim()
    elif args.bert_model_name:
        # Note: incompatible with other embedders, but logic in preprocess.py
        # should prevent these from being enabled anyway.
        from .bert.utils import BertEmbedderModule
        log.info(
            f"Using BERT model ({args.bert_model_name}); skipping other embedders.")
        cove_layer = None
        # Set PYTORCH_PRETRAINED_BERT_CACHE environment variable to an existing
        # cache; see
        # https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/file_utils.py
        bert_cache_dir = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE",
                                   os.path.join(args.exp_dir, "bert_cache"))
        maybe_make_dir(bert_cache_dir)
        embedder = BertEmbedderModule(args, cache_dir=bert_cache_dir)
        d_emb = embedder.get_output_dim()
    else:
        # Default case, used for ELMo, CoVe, word embeddings, etc.
        d_emb, embedder, cove_layer = build_embeddings(args, vocab,
                                                       tasks, pretrained_embs)
    d_sent_input = args.d_hid

    sent_encoder = build_sent_encoder(
        args, vocab, d_emb, tasks, embedder, cove_layer)

    sent_encoder, d_sent_output = build_sent_encoder(
        args, vocab, d_emb, tasks, embedder, cove_layer)
    # d_task_input is the input dimension of the task-specific module
    # set skip_emb = 1 if you want to concatenate the encoder input with encoder output to pass
    # into task specific module.
    d_task_input = d_sent_output + (args.skip_embs * d_emb)

    # Build model and classifiers
    model = MultiTaskModel(args, sent_encoder, vocab)
    build_task_modules(
        args,
        tasks,
        model,
        d_task_input,
        d_emb,
        embedder,
        vocab)
    model = model.cuda() if args.cuda >= 0 else model
    log.info(model)
    param_count = 0
    trainable_param_count = 0
    for name, param in model.named_parameters():
        param_count += np.prod(param.size())
        if param.requires_grad:
            trainable_param_count += np.prod(param.size())
            log.info(">> Trainable param %s: %s = %d", name,
                     str(param.size()), np.prod(param.size()))
    log.info(
        "Total number of parameters: {ct:d} ({ct:g})".format(
            ct=param_count))
    log.info("Number of trainable parameters: {ct:d} ({ct:g})".format(
        ct=trainable_param_count))
    return model


def get_task_whitelist(args):
    """Filters tasks so that we only build models that we will use, meaning we only
    build models for train tasks and for classifiers of eval tasks"""
    eval_task_names = parse_task_list_arg(args.target_tasks)
    eval_clf_names = []
    for task_name in eval_task_names:
        override_clf = config.get_task_attr(args, task_name, 'use_classifier')
        if override_clf == 'none' or override_clf is None:
            eval_clf_names.append(task_name)
        else:
            eval_clf_names.append(override_clf)
    train_task_names = parse_task_list_arg(args.pretrain_tasks)
    log.info("Whitelisting train tasks=%s, eval_clf_tasks=%s",
             str(train_task_names), str(eval_clf_names))
    return train_task_names, eval_clf_names


def build_embeddings(args, vocab, tasks, pretrained_embs=None):
    ''' Build embeddings according to options in args '''
    d_emb, d_char = 0, args.d_char

    token_embedders = {}
    # Word embeddings
    n_token_vocab = vocab.get_vocab_size('tokens')
    if args.word_embs != 'none':
        if args.word_embs in ['glove',
                              'fastText'] and pretrained_embs is not None:
            word_embs = pretrained_embs
            assert word_embs.size()[0] == n_token_vocab
            d_word = word_embs.size()[1]
            log.info("\tUsing pre-trained word embeddings: %s",
                     str(word_embs.size()))
        elif args.word_embs == "scratch":
            log.info("\tTraining word embeddings from scratch.")
            d_word = args.d_word
            word_embs = nn.Embedding(n_token_vocab, d_word).weight
        else:
            raise Exception(
                'Not a valid type of word emb. Set to none for elmo.')

        embeddings = Embedding(num_embeddings=n_token_vocab, embedding_dim=d_word,
                               weight=word_embs, trainable=(
                                   args.embeddings_train == 1),
                               padding_index=vocab.get_token_index('@@PADDING@@'))
        token_embedders["words"] = embeddings
        d_emb += d_word
    else:
        embeddings = None
        log.info("\tNot using word embeddings!")

    # Handle cove
    cove_layer = None
    if args.cove:
        assert embeddings is not None
        assert args.word_embs == "glove", "CoVe requires GloVe embeddings."
        assert d_word == 300, "CoVe expects 300-dimensional GloVe embeddings."
        try:
            from .modules.cove.cove import MTLSTM as cove_lstm
            # Have CoVe do an internal GloVe lookup, but don't add residual.
            # We'll do this manually in modules.py; see
            # SentenceEncoder.forward().
            cove_layer = cove_lstm(n_vocab=n_token_vocab,
                                   vectors=embeddings.weight.data)
            # Control whether CoVe is trainable.
            for param in cove_layer.parameters():
                param.requires_grad = bool(args.cove_fine_tune)
            d_emb += 600  # 300 x 2 for biLSTM activations
            log.info("\tUsing CoVe embeddings!")
        except ImportError as e:
            log.info("Failed to import CoVe!")
            raise e

    # Character embeddings
    if args.char_embs:
        log.info("\tUsing character embeddings!")
        char_embeddings = Embedding(vocab.get_vocab_size('chars'), d_char)
        filter_sizes = tuple([int(i)
                              for i in args.char_filter_sizes.split(',')])
        char_encoder = CnnEncoder(d_char, num_filters=args.n_char_filters,
                                  ngram_filter_sizes=filter_sizes,
                                  output_dim=d_char)
        char_embedder = TokenCharactersEncoder(char_embeddings, char_encoder,
                                               dropout=args.dropout_embs)
        d_emb += d_char
        token_embedders["chars"] = char_embedder
    else:
        log.info("\tNot using character embeddings!")

    # If we want separate ELMo scalar weights (a different ELMo representation for each classifier,
    # then we need count and reliably map each classifier to an index used by
    # allennlp internal ELMo.
    if args.sep_embs_for_skip:
        # Determine a deterministic list of classifier names to use for each
        # task.
        classifiers = sorted(set(map(lambda x: x._classifier_name, tasks)))
        # Reload existing classifier map, if it exists.
        classifier_save_path = args.run_dir + "/classifier_task_map.json"
        if os.path.isfile(classifier_save_path):
            loaded_classifiers = json.load(
                open(args.run_dir + "/classifier_task_map.json", 'r'))
        else:
            # No file exists, so assuming we are just starting to pretrain. If pretrain is to be
            # skipped, then there's a way to bypass this assertion by explicitly allowing for a missing
            # classiifer task map.
            assert_for_log(args.do_pretrain or args.allow_missing_task_map,
                           "Error: {} should already exist.".format(classifier_save_path))
            if args.allow_missing_task_map:
                log.warning("Warning: classifier task map not found in model"
                            " directory. Creating a new one from scratch.")
            # default is always @pretrain@
            loaded_classifiers = {"@pretrain@": 0}
        # Add the new tasks and update map, keeping the internal ELMo index
        # consistent.
        max_number_classifiers = max(loaded_classifiers.values())
        offset = 1
        for classifier in classifiers:
            if classifier not in loaded_classifiers:
                loaded_classifiers[classifier] = max_number_classifiers + offset
                offset += 1
        log.info("Classifiers:{}".format(loaded_classifiers))
        open(classifier_save_path, 'w+').write(json.dumps(loaded_classifiers))
        # Every index in classifiers needs to correspond to a valid ELMo output
        # representation.
        num_reps = 1 + max(loaded_classifiers.values())
    else:
        # All tasks share the same scalars.
        # Not used if self.elmo_chars_only = 1 (i.e. no elmo)
        loaded_classifiers = {"@pretrain@": 0}
        num_reps = 1
    if args.elmo:
        log.info("Loading ELMo from files:")
        log.info("ELMO_OPT_PATH = %s", ELMO_OPT_PATH)
        if args.elmo_chars_only:
            log.info("\tUsing ELMo character CNN only!")
            log.info("ELMO_WEIGHTS_PATH = %s", ELMO_WEIGHTS_PATH)
            elmo_embedder = ElmoCharacterEncoder(options_file=ELMO_OPT_PATH,
                                                 weight_file=ELMO_WEIGHTS_PATH,
                                                 requires_grad=False)
            d_emb += 512
        else:
            log.info("\tUsing full ELMo! (separate scalars/task)")
            if args.elmo_weight_file_path != 'none':
                assert os.path.exists(args.elmo_weight_file_path), "ELMo weight file path \"" + \
                    args.elmo_weight_file_path + "\" does not exist."
                weight_file = args.elmo_weight_file_path
            else:
                weight_file = ELMO_WEIGHTS_PATH
            log.info("ELMO_WEIGHTS_PATH = %s", weight_file)
            elmo_embedder = ElmoTokenEmbedderWrapper(
                options_file=ELMO_OPT_PATH,
                weight_file=weight_file,
                num_output_representations=num_reps,
                # Dropout is added by the sentence encoder later.
                dropout=0.)
            d_emb += 1024

        token_embedders["elmo"] = elmo_embedder

    # Wrap ELMo and other embedders, and concatenates the resulting
    # representations alone the last (vector) dimension.
    embedder = ElmoTextFieldEmbedder(token_embedders, loaded_classifiers,
                                     elmo_chars_only=args.elmo_chars_only,
                                     sep_embs_for_skip=args.sep_embs_for_skip)

    assert d_emb, "You turned off all the embeddings, ya goof!"
    return d_emb, embedder, cove_layer


def build_task_modules(args, tasks, model, d_sent, d_emb, embedder, vocab):
    """
        This function gets the task-specific parameters and builds
        the task-specific modules.
    """
    if args.is_probing_task:
        # TODO: move this logic to preprocess.py;
        # current implementation reloads MNLI data, which is slow.
        train_task_whitelist, eval_task_whitelist = get_task_whitelist(args)
        tasks_to_build, _, _ = get_tasks(train_task_whitelist,
                                         eval_task_whitelist,
                                         args.max_seq_len,
                                         path=args.data_dir,
                                         scratch_path=args.exp_dir)
    else:
        tasks_to_build = tasks

    # Attach task-specific params.
    for task in set(tasks + tasks_to_build):
        task_params = get_task_specific_params(args, task.name)
        log.info("\tTask '%s' params: %s", task.name,
                 json.dumps(task_params.as_dict(), indent=2))
        # Store task-specific params in case we want to access later
        setattr(model, '%s_task_params' % task.name, task_params)

    # Actually construct modules.
    for task in tasks_to_build:
        # If the name of the task is different than the classifier it should use
        # then skip the module creation.
        if task.name != model._get_task_params(
                task.name).get('use_classifier', task.name):
            log.info(
                "Name of the task is different than the classifier it should use")
            continue
        build_task_specific_modules(
            task, model, d_sent, d_emb, vocab, embedder, args)


def build_task_specific_modules(
        task, model, d_sent, d_emb, vocab, embedder, args):
    ''' Build task-specific components for a task and add them to model.
        These include decoders, linear layers for linear models.
     '''
    task_params = model._get_task_params(task.name)
    if isinstance(task, SingleClassificationTask):
        module = build_single_sentence_module(task=task, d_inp=d_sent,
                                              use_bert=model.use_bert, params=task_params)
        setattr(model, '%s_mdl' % task.name, module)
    elif isinstance(task, (PairClassificationTask, PairRegressionTask, PairOrdinalRegressionTask)):
        module = build_pair_sentence_module(task, d_sent, model=model, params=task_params)
        setattr(model, '%s_mdl' % task.name, module)
    elif isinstance(task, LanguageModelingTask):
        d_sent = args.d_hid + (args.skip_embs * d_emb)
        hid2voc = build_lm(task, d_sent, args)
        setattr(model, '%s_hid2voc' % task.name, hid2voc)
    elif isinstance(task, TaggingTask):
        hid2tag = build_tagger(task, d_sent, task.num_tags)
        setattr(model, '%s_mdl' % task.name, hid2tag)
    elif isinstance(task, EdgeProbingTask):
        module = EdgeClassifierModule(task, d_sent, task_params)
        setattr(model, '%s_mdl' % task.name, module)
    elif isinstance(task, (RedditSeq2SeqTask, Wiki103Seq2SeqTask)):
        log.info("using {} attention".format(args.s2s['attention']))
        decoder_params = Params({'input_dim': d_sent,
                                 'target_embedding_dim': 300,
                                 'decoder_hidden_size': args.s2s['d_hid_dec'],
                                 'output_proj_input_dim': args.s2s['output_proj_input_dim'],
                                 'max_decoding_steps': args.max_seq_len,
                                 'target_namespace': 'tokens',
                                 'attention': args.s2s['attention'],
                                 'dropout': args.dropout,
                                 'scheduled_sampling_ratio': 0.0})
        decoder = Seq2SeqDecoder(vocab, **decoder_params)
        setattr(model, '%s_decoder' % task.name, decoder)
    elif isinstance(task, MTTask):
        log.info("using {} attention".format(args.s2s['attention']))
        decoder_params = Params({'input_dim': d_sent,
                                 'target_embedding_dim': 300,
                                 'decoder_hidden_size': args.s2s['d_hid_dec'],
                                 'output_proj_input_dim': args.s2s['output_proj_input_dim'],
                                 'max_decoding_steps': args.max_seq_len,
                                 'target_namespace': task._label_namespace if hasattr(task,
                                                                                      '_label_namespace') else 'targets',
                                 'attention': args.s2s['attention'],
                                 'dropout': args.dropout,
                                 'scheduled_sampling_ratio': 0.0})
        decoder = Seq2SeqDecoder(vocab, **decoder_params)
        setattr(model, '%s_decoder' % task.name, decoder)

    elif isinstance(task, SequenceGenerationTask):
        decoder, hid2voc = build_decoder(task, d_sent, vocab, embedder, args)
        setattr(model, '%s_decoder' % task.name, decoder)
        setattr(model, '%s_hid2voc' % task.name, hid2voc)

    elif isinstance(task, (GroundedTask, GroundedSWTask)):
        task.img_encoder = CNNEncoder(model_name='resnet', path=task.path)
        pooler = build_image_sent_module(task, d_sent, task_params)
        setattr(model, '%s_mdl' % task.name, pooler)
    elif isinstance(task, RankingTask):
        pooler, dnn_ResponseModel = build_reddit_module(
            task, d_sent, task_params)
        setattr(model, '%s_mdl' % task.name, pooler)
        setattr(model, '%s_Response_mdl' % task.name, dnn_ResponseModel)
    else:
        raise ValueError("Module not found for %s" % task.name)


def get_task_specific_params(args, task_name):
    ''' Search args for parameters specific to task.
    Args:
        args: main-program args, a config.Params object
        task_name: (string)
    Returns:
        AllenNLP Params object of task-specific params.
    '''
    def _get_task_attr(attr_name): return config.get_task_attr(args, task_name,
                                                               attr_name)
    params = {}
    params['cls_type'] = _get_task_attr("classifier")
    params['d_hid'] = _get_task_attr("classifier_hid_dim")
    params['d_proj'] = _get_task_attr("d_proj")
    params['shared_pair_attn'] = args.shared_pair_attn
    if args.shared_pair_attn:
        params['attn'] = args.pair_attn
        params['d_hid_attn'] = args.d_hid_attn
        params['dropout'] = args.classifier_dropout
    else:
        params['attn'] = _get_task_attr("pair_attn")
        params['d_hid_attn'] = _get_task_attr("d_hid_attn")
        params['dropout'] = _get_task_attr("classifier_dropout")

    # Used for edge probing. Other tasks can safely ignore.
    params['cls_loss_fn'] = _get_task_attr("classifier_loss_fn")
    params['cls_span_pooling'] = _get_task_attr("classifier_span_pooling")
    params['edgeprobe_cnn_context'] = _get_task_attr("edgeprobe_cnn_context")

    # For NLI probing tasks, might want to use a classifier trained on
    # something else (typically 'mnli').
    cls_task_name = _get_task_attr("use_classifier")
    # default to this task
    params['use_classifier'] = cls_task_name or task_name

    return Params(params)


def build_reddit_module(task, d_inp, params):
    ''' Build a single classifier '''
    pooler = Pooler(project=True, d_inp=d_inp, d_proj=params['d_proj'])
    dnn_ResponseModel = nn.Sequential(nn.Linear(params['d_proj'], params['d_proj']),
                                      nn.Tanh(), nn.Linear(params['d_proj'], params['d_proj']))
    return pooler, dnn_ResponseModel


def build_image_sent_module(task, d_inp, params):
    pooler = Pooler(project=True, d_inp=d_inp, d_proj=params['d_proj'])
    return pooler


def build_single_sentence_module(task, d_inp: int, use_bert: bool, params: Params):
    ''' Build a single sentence classifier

    args:
        - task (Task): task object, used to get the number of output classes
        - d_inp (int): input dimension to the module, needed for optional linear projection
        - use_bert (bool): if using BERT, extract the first vector from the inputted
            sequence, rather than max pooling. We do this for BERT specifically to follow
            the convention set in the paper (Devlin et al., 2019).
        - params (Params): Params object with task-specific parameters

    returns:
        - SingleClassifier (nn.Module): single-sentence classifier consisting of
            (optional) a linear projection, pooling, and an MLP classifier
    '''
    pool_type = "first" if use_bert else "max"
    pooler = Pooler(project=not use_bert, d_inp=d_inp, d_proj=params['d_proj'], pool_type=pool_type)
    d_out = d_inp if use_bert else params["d_proj"]
    classifier = Classifier.from_params(d_out, task.n_classes, params)
    module = SingleClassifier(pooler, classifier)
    return module


def build_pair_sentence_module(task, d_inp, model, params):
    ''' Build a pair classifier, shared if necessary '''

    def build_pair_attn(d_in, d_hid_attn):
        ''' Build the pair model '''
        d_inp_model = 2 * d_in
        modeling_layer = s2s_e.by_name('lstm').from_params(
            Params({'input_size': d_inp_model, 'hidden_size': d_hid_attn,
                    'num_layers': 1, 'bidirectional': True}))
        pair_attn = AttnPairEncoder(model.vocab, modeling_layer, dropout=params["dropout"])
        return pair_attn

    # Build the "pooler", which does pools a variable length sequence
    #   possibly with a projection layer beforehand
    if params["attn"] and not model.use_bert:
        pooler = Pooler(project=False, d_inp=params["d_hid_attn"], d_proj=params["d_hid_attn"])
        d_out = params["d_hid_attn"] * 2
    else:
        pool_type = "first" if model.use_bert else "max"
        pooler = Pooler(project=not model.use_bert, d_inp=d_inp, d_proj=params["d_proj"], pool_type=pool_type)
        d_out = d_inp if model.use_bert else params["d_proj"]


    # Build an attention module if necessary
    if params["shared_pair_attn"] and params["attn"] and not model.use_bert: # shared attn
        if not hasattr(model, "pair_attn"):
            pair_attn = build_pair_attn(d_inp, params["d_hid_attn"])
            model.pair_attn = pair_attn
        else:
            pair_attn = model.pair_attn
    elif params["attn"] and not model.use_bert: # non-shared attn
        pair_attn = build_pair_attn(d_inp, params["d_hid_attn"])
    else:  # no attn
        pair_attn = None

    # Build the classifier
    n_classes = task.n_classes if hasattr(task, 'n_classes') else 1
    if model.use_bert:
        # BERT handles pair tasks by concatenating the inputs and classifying the joined
        # sequence, so we use a single sentence classifier
        classifier = Classifier.from_params(d_out, n_classes, params)
        module = SingleClassifier(pooler, classifier)
    else:
        classifier = Classifier.from_params(4 * d_out, n_classes, params)
        module = PairClassifier(pooler, classifier, pair_attn)
    return module


def build_lm(task, d_inp, args):
    ''' Build LM components (just map hidden states to vocab logits) '''
    hid2voc = nn.Linear(d_inp, args.max_word_v_size)
    return hid2voc


def build_tagger(task, d_inp, out_dim):
    ''' Build tagger components. '''
    hid2tag = nn.Linear(d_inp, out_dim)
    return hid2tag


def build_decoder(task, d_inp, vocab, embedder, args):
    ''' Build a task specific decoder '''
    rnn = s2s_e.by_name('lstm').from_params(
        Params({'input_size': embedder.get_output_dim(),
                'hidden_size': args.s2s['d_hid_dec'],
                'num_layers': args.s2s['n_layers_dec'], 'bidirectional': False}))
    decoder = SentenceEncoder(vocab, embedder, 0, rnn)
    hid2voc = nn.Linear(args.s2s['d_hid_dec'], args.max_word_v_size)
    return decoder, hid2voc


class MultiTaskModel(nn.Module):
    '''
    Giant model with task-specific components and a shared word and sentence encoder.
    This class samples the tasks passed in pretrained_tasks, and adds task specific components
    to the model.
    '''

    def __init__(self, args, sent_encoder, vocab):
        ''' Args: sentence encoder '''
        super(MultiTaskModel, self).__init__()
        self.sent_encoder = sent_encoder
        self.vocab = vocab
        self.utilization = Average() if args.track_batch_utilization else None
        self.elmo = args.elmo and not args.elmo_chars_only
        self.use_bert = bool(args.bert_model_name)
        self.sep_embs_for_skip = args.sep_embs_for_skip

    def forward(self, task, batch, predict=False):
        '''
        Pass inputs to correct forward pass
        Args:
            - task (tasks.Task): task for which batch is drawn
            - batch (Dict[str:Dict[str:Tensor]]): dictionary of (field, indexing) pairs,
                where indexing is a dict of the index namespace and the actual indices.
            - predict (Bool): passed to task specific forward(). If true, forward()
                should return predictions.
        Returns:
            - out: dictionary containing task outputs and loss if label was in batch
        '''
        if self.utilization is not None:
            if 'input1' in batch:
                self.utilization(get_batch_utilization(batch['input1']))
            elif 'input' in batch:
                self.utilization(get_batch_utilization(batch['input']))
        if isinstance(task, SingleClassificationTask):
            out = self._single_sentence_forward(batch, task, predict)
        elif isinstance(task, MultiNLIDiagnosticTask):
            out = self._pair_sentence_MNLI_diagnostic_forward(
                batch, task, predict)
        elif isinstance(task, (PairClassificationTask, PairRegressionTask,
                               PairOrdinalRegressionTask)):
            if task.name in [
                'wiki103_classif',
                'reddit_pair_classif',
                'reddit_pair_classif_mini',
                'reddit_pair_classif_3.4G',
                'mt_pair_classif',
                    'mt_pair_classif_mini']:
                out = self._positive_pair_sentence_forward(
                    batch, task, predict)
            else:
                out = self._pair_sentence_forward(batch, task, predict)
        elif isinstance(task, LanguageModelingTask):
            out = self._lm_forward(batch, task, predict)
        elif isinstance(task, TaggingTask):
            out = self._tagger_forward(batch, task, predict)
        elif isinstance(task, EdgeProbingTask):
            # Just get embeddings and invoke task module.
            sent_embs, sent_mask = self.sent_encoder(batch['input1'], task)
            module = getattr(self, "%s_mdl" % task.name)
            out = module.forward(batch, sent_embs, sent_mask,
                                 task, predict)
        elif isinstance(task, SequenceGenerationTask):
            out = self._seq_gen_forward(batch, task, predict)
        elif isinstance(task, (GroundedTask, GroundedSWTask)):
            out = self._grounded_ranking_bce_forward(batch, task, predict)
        elif isinstance(task, RankingTask):
            out = self._ranking_forward(batch, task, predict)
        else:
            raise ValueError("Task-specific components not found!")
        return out

    def _get_task_params(self, task_name):
        """ Get task-specific Params, as set in build_module(). """
        return getattr(self, "%s_task_params" % task_name)

    def _get_classifier(self, task):
        """ Get task-specific classifier, as set in build_module(). """
        # TODO: replace this logic with task._classifier_name?
        task_params = self._get_task_params(task.name)
        use_clf = task_params['use_classifier']
        if use_clf in [None, "", "none"]:
            use_clf = task.name  # default if not set
        return getattr(self, "%s_mdl" % use_clf)

    def _single_sentence_forward(self, batch, task, predict):
        out = {}

        # embed the sentence
        sent_embs, sent_mask = self.sent_encoder(batch['input1'], task)
        # pass to a task specific classifier
        classifier = self._get_classifier(task)
        logits = classifier(sent_embs, sent_mask)
        out['logits'] = logits
        out['n_exs'] = get_batch_size(batch)

        if 'labels' in batch:  # means we should compute loss
            if batch['labels'].dim() == 0:
                labels = batch['labels'].unsqueeze(0)
            elif batch['labels'].dim() == 1:
                labels = batch['labels']
            else:
                labels = batch['labels'].squeeze(-1)
            out['loss'] = F.cross_entropy(logits, labels)
            tagmask = batch.get('tagmask', None)
            task.update_metrics(logits, labels, tagmask=tagmask)

        if predict:
            if isinstance(task, RegressionTask):
                if logits.ndimension() > 1:
                    assert logits.ndimension() == 2 and logits[-1] == 1, \
                        "Invalid regression prediction dimensions!"
                    logits = logits.squeeze(-1)
                out['preds'] = logits
            else:
                _, out['preds'] = logits.max(dim=1)
        return out

    def _pair_sentence_MNLI_diagnostic_forward(self, batch, task, predict):
        out = {}

        # embed the sentence
        classifier = self._get_classifier(task)
        if self.use_bert:
            sent, mask = self.sent_encoder(batch['inputs'], task)
            logits = classifier(sent, mask)
        else:
            sent1, mask1 = self.sent_encoder(batch['input1'], task)
            sent2, mask2 = self.sent_encoder(batch['input2'], task)
            logits = classifier(sent1, sent2, mask1, mask2)
        out['logits'] = logits
        out['n_exs'] = get_batch_size(batch)

        labels = batch['labels'].squeeze(-1)
        out['loss'] = F.cross_entropy(logits, labels)
        _, predicted = logits.max(dim=1)
        if 'labels' in batch:
            if batch['labels'].dim() == 0:
                labels = batch['labels'].unsqueeze(0)
            elif batch['labels'].dim() == 1:
                labels = batch['labels']
            else:
                labels = batch['labels'].squeeze(-1)
            out['loss'] = F.cross_entropy(logits, labels)
            task.update_diagnostic_metrics(predicted, labels, batch)

        if predict:
            out['preds'] = predicted
        return out

    def _positive_pair_sentence_forward(self, batch, task, predict):
        ''' forward function written specially for cases where we have only +ve pairs in input data
            -ve pairs are created by rotating either sent1 or sent2.
            Ex: [1,2,3,4] after rotation by 2 positions [3,4,1,2]
            Assumption is each example in sent1 has only one corresponding example in sent2 which is +ve
            So rotating sent1/sent2 and pairing with sent2/sent1 is one way to obtain -ve pairs
        '''
        out = {}

        assert_for_log(not self.use_bert, "BERT is currently not supported for negative sampling!")
        # issue with using BERT here is that input1 and input2 are padded already
        # so concatenating to get negative samples is fairly annoying

        # embed the sentence
        sent1, mask1 = self.sent_encoder(batch['input1'], task)
        sent2, mask2 = self.sent_encoder(batch['input2'], task)
        classifier = self._get_classifier(task)

        # Negative pairs are created by rotating sent2
        # Note that we need to rotate corresponding mask also. *_new contain
        # positive and negative pairs
        sent1_new = torch.cat([sent1, sent1], 0)
        mask1_new = torch.cat([mask1, mask1], 0)
        sent2_new = torch.cat(
            [sent2, torch.cat([sent2[2:], sent2[0:2]], 0)], 0)
        mask2_new = torch.cat(
            [mask2, torch.cat([mask2[2:], mask2[0:2]], 0)], 0)
        logits = classifier(sent1_new, sent2_new, mask1_new, mask2_new)
        out['logits'] = logits
        out['n_exs'] = len(sent1_new)
        labels = torch.cat([torch.ones(len(sent1)), torch.zeros(len(sent1))])
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        out['loss'] = F.cross_entropy(logits, labels)
        tagmask = batch.get('tagmask', None)
        task.update_metrics(logits, labels, tagmask=tagmask)

        if predict:
            if isinstance(task, RegressionTask):
                if logits.ndimension() > 1:
                    assert logits.ndimension() == 2 and logits[-1] == 1, \
                        "Invalid regression prediction dimensions!"
                    logits = logits.squeeze(-1)
                out['preds'] = logits
            else:
                _, out['preds'] = logits.max(dim=1)
        return out

    def _pair_sentence_forward(self, batch, task, predict):
        out = {}

        # embed the sentence
        classifier = self._get_classifier(task)
        if self.use_bert:
            sent, mask = self.sent_encoder(batch['inputs'], task)
            logits = classifier(sent, mask)
        else:
            sent1, mask1 = self.sent_encoder(batch['input1'], task)
            sent2, mask2 = self.sent_encoder(batch['input2'], task)
            logits = classifier(sent1, sent2, mask1, mask2)
        out['logits'] = logits
        out['n_exs'] = get_batch_size(batch)
        tagmask = batch.get('tagmask', None)
        if 'labels' in batch:
            labels = batch['labels']
            labels = labels.squeeze(-1) if len(labels.size()) > 1 else labels
            if isinstance(task, JOCITask):
                logits = logits.squeeze(-1) if len(logits.size()
                                                   ) > 1 else logits
                out['loss'] = F.mse_loss(logits, labels)
                logits_np = logits.data.cpu().numpy()
                labels_np = labels.data.cpu().numpy()
                task.scorer1(mean_squared_error(logits_np, labels_np))
                task.scorer2(logits_np, labels_np)
            elif isinstance(task, STSBTask):
                logits = logits.squeeze(-1) if len(logits.size()
                                                   ) > 1 else logits
                out['loss'] = F.mse_loss(logits, labels)
                task.update_metrics(logits.data.cpu().numpy(), labels.data.cpu().numpy(), tagmask=tagmask)
            else:
                out['loss'] = F.cross_entropy(logits, labels)
                task.update_metrics(logits, labels, tagmask=tagmask)

        if predict:
            if isinstance(task, RegressionTask):
                if logits.ndimension() > 1:
                    assert logits.ndimension() == 2 and logits[-1] == 1, \
                        "Invalid regression prediction dimensions!"
                    logits = logits.squeeze(-1)
                out['preds'] = logits
            else:
                _, out['preds'] = logits.max(dim=1)
        return out

    def _ranking_forward(self, batch, task, predict):
        ''' For caption and image ranking. This implementation is intended for Reddit
            This implementation assumes only positive pairs exist in input data.
            Negative pairs are created within batch.
        '''
        out = {}
        # feed forwarding inputs through sentence encoders
        sent1, mask1 = self.sent_encoder(batch['input1'], task)
        sent2, mask2 = self.sent_encoder(batch['input2'], task)
        # pooler for both Input and Response
        sent_pooler = getattr(self, "%s_mdl" % task.name)
        sent_dnn = getattr(
            self, "%s_Response_mdl" %
            task.name)  # dnn for Response
        sent1_rep = sent_pooler(sent1, mask1)
        sent2_rep_pool = sent_pooler(sent2, mask2)
        sent2_rep = sent_dnn(sent2_rep_pool)

        cos_simi = torch.mm(sent1_rep, sent2_rep.transpose(0, 1))
        if task.name == 'reddit_softmax':
            cos_simi_backward = cos_simi.transpose(0, 1)
            labels = torch.arange(len(cos_simi), dtype=torch.long).cuda()

            total_loss = torch.nn.CrossEntropyLoss()(cos_simi, labels)  # one-way loss
            total_loss_rev = torch.nn.CrossEntropyLoss()(
                cos_simi_backward, labels)  # reverse
            out['loss'] = total_loss + total_loss_rev

            pred = torch.nn.Softmax(dim=1)(cos_simi)
            pred = torch.argmax(pred, dim=1)
        else:
            labels = torch.eye(len(cos_simi))

            # balancing pairs: #positive_pairs = batch_size, #negative_pairs =
            # batch_size-1
            cos_simi_pos = torch.diag(cos_simi)
            cos_simi_neg = torch.diag(cos_simi, diagonal=1)
            cos_simi = torch.cat([cos_simi_pos, cos_simi_neg], dim=0)
            labels_pos = torch.diag(labels)
            labels_neg = torch.diag(labels, diagonal=1)
            labels = torch.cat([labels_pos, labels_neg], dim=0)
            labels = labels.cuda()
            total_loss = torch.nn.BCEWithLogitsLoss()(cos_simi, labels)
            out['loss'] = total_loss

            pred = torch.sigmoid(cos_simi).round()

        total_correct = torch.sum(pred == labels)
        batch_acc = total_correct.item() / len(labels)
        out["n_exs"] = len(labels)
        task.scorer1(batch_acc)
        return out

    def _seq_gen_forward(self, batch, task, predict):
        ''' For variational autoencoder '''
        out = {}
        sent, sent_mask = self.sent_encoder(batch['inputs'], task)
        out['n_exs'] = get_batch_size(batch)

        if isinstance(task, (MTTask, RedditSeq2SeqTask)):
            decoder = getattr(self, "%s_decoder" % task.name)
            out.update(decoder.forward(sent, sent_mask, batch['targs']))
            task.scorer1(out['loss'].item())

        if 'targs' in batch:
            pass

        if predict:
            pass

        return out

    def _tagger_forward(self, batch, task, predict):
        ''' For sequence tagging '''
        out = {}
        b_size, seq_len, _ = batch['inputs']['elmo'].size()
        seq_len -= 2
        sent_encoder = self.sent_encoder
        out['n_exs'] = get_batch_size(batch)
        if not isinstance(sent_encoder, BiLMEncoder):
            sent, mask = sent_encoder(batch['inputs'], task)
            sent = sent.masked_fill(1 - mask.byte(), 0)  # avoid NaNs
            sent = sent[:, 1:-1, :]
            hid2tag = self._get_classifier(task)
            logits = hid2tag(sent)
            logits = logits.view(b_size * seq_len, -1)
            out['logits'] = logits
            targs = batch['targs']['words'][:, :seq_len].contiguous().view(-1)

        pad_idx = self.vocab.get_token_index(self.vocab._padding_token)
        out['loss'] = F.cross_entropy(logits, targs, ignore_index=pad_idx)
        task.scorer1(logits, targs)
        return out

    def _lm_forward(self, batch, task, predict):
        """Forward pass for LM model
        Args:
            batch: indexed input data
            task: (Task obejct)
            predict: (boolean) predict mode (not supported)
        return:
            out: (dict)
                - 'logits': output layer, dimension: [batchSize * timeSteps * 2, outputDim]
                            first half: [:batchSize*timeSteps, outputDim] is output layer from forward layer
                            second half: [batchSize*timeSteps:, outputDim] is output layer from backward layer
                - 'loss': size average CE loss
        """
        out = {}
        sent_encoder = self.sent_encoder
        assert_for_log(isinstance(sent_encoder._phrase_layer, BiLMEncoder),
                       "Not using LM for language modeling task!")
        assert_for_log('targs' in batch and 'words' in batch['targs'],
                       "Batch missing target words!")
        pad_idx = self.vocab.get_token_index(
            self.vocab._padding_token, 'tokens')
        b_size, seq_len = batch['targs']['words'].size()
        n_pad = batch['targs']['words'].eq(pad_idx).sum().item()
        out['n_exs'] = (b_size * seq_len - n_pad) * 2

        sent, mask = sent_encoder(batch['input'], task)
        sent = sent.masked_fill(1 - mask.byte(), 0)  # avoid NaNs

        # Split encoder outputs by direction
        split = int(self.sent_encoder._phrase_layer.get_output_dim() / 2)
        fwd, bwd = sent[:, :, :split], sent[:, :, split:split * 2]
        if split * 2 < sent.size(2):  # skip embeddings
            out_embs = sent[:, :, split * 2:]
            fwd = torch.cat([fwd, out_embs], dim=2)
            bwd = torch.cat([bwd, out_embs], dim=2)

        # Forward and backward logits and targs
        hid2voc = getattr(self, "%s_hid2voc" % task.name)
        logits_fwd = hid2voc(fwd).view(b_size * seq_len, -1)
        logits_bwd = hid2voc(bwd).view(b_size * seq_len, -1)
        logits = torch.cat([logits_fwd, logits_bwd], dim=0)
        out['logits'] = logits
        trg_fwd = batch['targs']['words'].view(-1)
        trg_bwd = batch['targs_b']['words'].view(-1)
        targs = torch.cat([trg_fwd, trg_bwd], dim=0)
        assert logits.size(0) == targs.size(
            0), "Number of logits and targets differ!"
        out['loss'] = F.cross_entropy(logits, targs, ignore_index=pad_idx)
        task.scorer1(out['loss'].item())
        if predict:
            pass
        return out

    def _grounded_forward(self, batch, task, predict):
        out, img_seq = {}, []
        sent_emb, sent_mask = self.sent_encoder(batch['input1'], task)
        batch_size = get_batch_size(batch)
        out['n_exs'] = batch_size
        sent_pooler = self._get_classifier(task)

        sent_rep = sent_pooler(sent_emb, sent_mask)

        ids = batch['ids'].cpu().squeeze(-1).data.numpy().tolist()

        for img_idx in ids:
            img_rep = task.img_encoder.forward(int(img_idx))[0]
            img_seq.append(torch.tensor(img_rep, dtype=torch.float32).cuda())

        loss = torch.autograd.Variable(torch.Tensor(1), requires_grad=True) + 0
        softmax = nn.Softmax(dim=0)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        acc = []

        loss_fn = nn.L1Loss()
        # contrastive against n samples (n = {2, 3}), temperature
        samples, temp = batch_size - 1, 0.001
        for sent_idx in range(batch_size):
            sent = sent_rep[sent_idx].reshape(1, -1).cuda()
            img = img_seq[sent_idx].reshape(1, -1).cuda()
            labels = [1] + [0] * (samples - 1)
            labels = torch.tensor(labels, dtype=torch.float32)
            mat = [cos(sent, img).cpu().data.numpy()[0]]
            for _ in range(len(mat), samples):
                r = sent_idx
                while (r == sent_idx):
                    r = np.random.randint(batch_size, size=(1, 1))[0][0]
                img = img_seq[r].reshape(1, -1).cuda()
                mat.append(cos(sent, img).cpu().data.numpy()[0])

            mat = torch.tensor(mat, dtype=torch.float32)
            dist = softmax(Variable(torch.tensor(mat, dtype=torch.float32)))

            max_idx = np.argmax(dist.data.numpy())
            loss.add(loss_fn(mat, labels))

            preds = [0] * samples
            preds[max_idx] = 1
            acc.append(1 if max_idx == 0 else 0)

        out['loss'] = loss
        task.scorer1(np.mean(acc))
        return out

    def _grounded_ranking_bce_forward(self, batch, task, predict):
        ''' Binary Cross Entropy Loss
            Create sentence, image representation.
        '''

        out, neg = {}, []
        sent_emb, sent_mask = self.sent_encoder(batch['input1'], task)
        batch_size = get_batch_size(batch)
        out['n_exs'] = batch_size
        sent_pooler = self._get_classifier(task)
        sent_rep = sent_pooler(sent_emb, sent_mask)
        loss_fn = nn.L1Loss()
        ids = batch['ids'].cpu().squeeze(-1).data.numpy().tolist()
        img_seq = []

        for img_idx in ids:
            img_rep = task.img_encoder.forward(int(img_idx))[0]
            img_seq.append(torch.tensor(img_rep, dtype=torch.float32).cuda())

        img_emb = torch.stack(img_seq, dim=0)
        sent1_rep = sent_rep
        sent2_rep = img_emb

        sent1_rep = F.normalize(sent1_rep, 2, 1)
        sent2_rep = F.normalize(sent2_rep, 2, 1)
        mat_mul = torch.mm(sent1_rep, torch.transpose(sent2_rep, 0, 1))
        labels = torch.eye(len(mat_mul))

        scale = 1 / (len(mat_mul) - 1) if len(mat_mul) > 1 else 1
        weights = scale * torch.ones(mat_mul.shape) - \
            (scale - 1) * torch.eye(len(mat_mul))
        weights = weights.view(-1).cuda()

        mat_mul = mat_mul.view(-1)
        labels = labels.view(-1).cuda()
        pred = torch.sigmoid(mat_mul).round()

        out['loss'] = loss_fn(mat_mul, labels)
        total_correct = torch.sum(pred == labels)
        batch_acc = total_correct.item() / len(labels)
        task.scorer1.__call__(batch_acc)

        return out

    def get_elmo_mixing_weights(self, tasks=[]):
        ''' Get elmo mixing weights from text_field_embedder. Gives warning when fails.
        args:
           - tasks (List[Task]): list of tasks that we want to get  ELMo scalars for.
        returns:
            - params Dict[str:float]: dictionary maybe layers to scalar params
        '''
        params = {}
        if self.elmo:
            if not self.sep_embs_for_skip:
                tasks = [None]
            else:
                tasks = [None] + tasks
            for task in tasks:
                if task:
                    params[task._classifier_name] = get_elmo_mixing_weights(
                        self.sent_encoder._text_field_embedder, task=task)
                else:
                    params["@pretrain@"] = get_elmo_mixing_weights(
                        self.sent_encoder._text_field_embedder, task=None)
        return params
