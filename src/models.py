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
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.similarity_functions import DotProductSimilarity
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder, \
                                              PytorchSeq2SeqWrapper
from allennlp.training.metrics import Average

from .allennlp_mods.elmo_text_field_embedder import ElmoTextFieldEmbedder, ElmoTokenEmbedderWrapper
from .utils import get_batch_utilization, get_elmo_mixing_weights
from . import config
from . import edge_probing
from . import beamsearch

from .tasks import STSBTask, CoLATask, SSTTask, \
    PairClassificationTask, SingleClassificationTask, \
    PairRegressionTask, RankingTask, \
    SequenceGenerationTask, LanguageModelingTask, \
    PairOrdinalRegressionTask, JOCITask, WeakGroundedTask, \
    GroundedTask, MTTask, RedditTask, Reddit_MTTask, Wiki103_Seq2Seq

from .tasks import STSBTask, CoLATask, \
    ClassificationTask, PairClassificationTask, SingleClassificationTask, \
    RegressionTask, PairRegressionTask, RankingTask, \
    SequenceGenerationTask, LanguageModelingTask, MTTask, \
    PairOrdinalRegressionTask, JOCITask, \
    WeakGroundedTask, GroundedTask, VAETask, \
    GroundedTask, TaggingTask, POSTaggingTask, CCGTaggingTask, \
    MultiNLIDiagnosticTask
from .tasks import EdgeProbingTask

from .modules import SentenceEncoder, BoWSentEncoder, \
    AttnPairEncoder, MaskedStackedSelfAttentionEncoder, \
    BiLMEncoder, ElmoCharacterEncoder, Classifier, Pooler, \
    SingleClassifier, PairClassifier, CNNEncoder, \
    NullPhraseLayer

from .utils import assert_for_log, get_batch_utilization, get_batch_size
from .preprocess import parse_task_list_arg, get_tasks
from .seq2seq_decoder import Seq2SeqDecoder


# Elmo stuff
# Look in $ELMO_SRC_DIR (e.g. /usr/share/jsalt/elmo) or download from web
ELMO_OPT_NAME = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHTS_NAME = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
ELMO_SRC_DIR = (os.getenv("ELMO_SRC_DIR") or
                "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/")
ELMO_OPT_PATH = os.path.join(ELMO_SRC_DIR, ELMO_OPT_NAME)
ELMO_WEIGHTS_PATH = os.path.join(ELMO_SRC_DIR, ELMO_WEIGHTS_NAME)


def build_model(args, vocab, pretrained_embs, tasks):
    '''Build model according to args '''

    # Build embeddings.
    d_emb, embedder, cove_emb = build_embeddings(args, vocab, tasks, pretrained_embs)
    d_sent = args.d_hid

    # Build single sentence encoder: the main component of interest
    # Need special handling for language modeling
    tfm_params = Params({'input_dim': d_emb, 'hidden_dim': args.d_hid,
                         'projection_dim': args.d_tproj,
                         'feedforward_hidden_dim': args.d_ff,
                         'num_layers': args.n_layers_enc,
                         'num_attention_heads': args.n_heads})
    rnn_params = Params({'input_size': d_emb, 'bidirectional': True,
                         'hidden_size': args.d_hid, 'num_layers': args.n_layers_enc})

    if sum([isinstance(task, LanguageModelingTask) for task in tasks]) or \
            args.sent_enc == 'bilm':
        assert_for_log(args.sent_enc in ['rnn', 'bilm'], "Only RNNLM supported!")
        if args.elmo:
            assert_for_log(args.elmo_chars_only, "LM with full ELMo not supported")
        bilm = BiLMEncoder(d_emb, args.d_hid, args.d_hid, args.n_layers_enc)
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       bilm, skip_embs=args.skip_embs,
                                       dropout=args.dropout,
                                       sep_embs_for_skip=args.sep_embs_for_skip,
                                       cove_layer=cove_emb)
        d_sent = 2 * args.d_hid
        log.info("Using BiLM architecture for shared encoder!")
    elif args.sent_enc == 'bow':
        sent_encoder = BoWSentEncoder(vocab, embedder)
        log.info("Using BoW architecture for shared encoder!")
        d_sent = d_emb
    elif args.sent_enc == 'rnn':
        sent_rnn = s2s_e.by_name('lstm').from_params(copy.deepcopy(rnn_params))
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       sent_rnn, skip_embs=args.skip_embs,
                                       dropout=args.dropout, sep_embs_for_skip=args.sep_embs_for_skip,
                                       cove_layer=cove_emb)
        d_sent = 2 * args.d_hid
        log.info("Using BiLSTM architecture for shared encoder!")
    elif args.sent_enc == 'transformer':
        transformer = StackedSelfAttentionEncoder.from_params(copy.deepcopy(tfm_params))
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       transformer, dropout=args.dropout,
                                       skip_embs=args.skip_embs, cove_layer=cove_emb,
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
                                       cove_layer=cove_emb)
        d_sent = 0  # skip connection added below
        log.info("No shared encoder (just using word embeddings)!")
    else:
        assert_for_log(False, "No valid sentence encoder specified.")

    d_sent += args.skip_embs * d_emb

    # Build model and classifiers
    model = MultiTaskModel(args, sent_encoder, vocab)

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
        # If the name of the task is different than the classifier it should use then skip the module creation.
        if task.name != model._get_task_params(task.name).get('use_classifier', task.name):
            continue
        build_module(task, model, d_sent, d_emb, vocab, embedder, args)
    model = model.cuda() if args.cuda >= 0 else model
    log.info(model)
    param_count = 0
    trainable_param_count = 0
    for name, param in model.named_parameters():
        param_count += np.prod(param.size())
        if param.requires_grad:
            trainable_param_count += np.prod(param.size())
    log.info("Total number of parameters: {}".format(param_count))
    log.info("Number of trainable parameters: {}".format(trainable_param_count))
    return model

def get_task_whitelist(args):
  """Filters tasks so that we only build models that we will use, meaning we only
  build models for train tasks and for classifiers of eval tasks"""
  eval_task_names = parse_task_list_arg(args.eval_tasks)
  eval_clf_names = []
  for task_name in eval_task_names:
    override_clf = config.get_task_attr(args, task_name, 'use_classifier')
    if override_clf == 'none'  or override_clf is None:
      eval_clf_names.append(task_name)
    else:
      eval_clf_names.append(override_clf)
  train_task_names = parse_task_list_arg(args.train_tasks)
  log.info("Whitelisting train tasks=%s, eval_clf_tasks=%s"%(str(train_task_names), str(eval_clf_names)))
  return train_task_names, eval_clf_names

def build_embeddings(args, vocab, tasks, pretrained_embs=None):
    ''' Build embeddings according to options in args '''
    d_emb, d_char = 0, args.d_char

    token_embedder = {}
    # Word embeddings
    if args.word_embs != 'none':
        if args.word_embs in ['glove', 'fastText'] and pretrained_embs is not None:
            log.info("\tUsing word embeddings from %s", args.word_embs_file)
            word_embs = pretrained_embs
            d_word = pretrained_embs.size()[-1]
        else:
            log.info("\tLearning word embeddings from scratch!")
            word_embs = None
            d_word = args.d_word

        embeddings = Embedding(vocab.get_vocab_size('tokens'), d_word,
                               weight=word_embs, trainable=False,
                               padding_index=vocab.get_token_index('@@PADDING@@'))
        token_embedder["words"] = embeddings
        d_emb += d_word
    else:
        log.info("\tNot using word embeddings!")

    # Handle cove
    if args.cove:
        sys.path.append(args.path_to_cove)
        try:
            from cove import MTLSTM as cove_lstm
            cove_emb = cove_lstm(n_vocab=vocab.get_vocab_size('tokens'),
                                 vectors=embeddings.weight.data)
            d_emb += 600
            log.info("\tUsing CoVe embeddings!")
        except ImportError:
            log.info("Failed to import CoVE!")
    else:
        cove_emb = None

    # Character embeddings
    if args.char_embs:
        log.info("\tUsing character embeddings!")
        char_embeddings = Embedding(vocab.get_vocab_size('chars'), d_char)
        filter_sizes = tuple([int(i) for i in args.char_filter_sizes.split(',')])
        char_encoder = CnnEncoder(d_char, num_filters=args.n_char_filters,
                                  ngram_filter_sizes=filter_sizes,
                                  output_dim=d_char)
        char_embedder = TokenCharactersEncoder(char_embeddings, char_encoder,
                                               dropout=args.dropout_embs)
        d_emb += d_char
        token_embedder["chars"] = char_embedder
    else:
        log.info("\tNot using character embeddings!")

    # Handle elmo
    if args.sep_embs_for_skip:
        # one representation per task
        reps = tasks
    else:
        # no unique rep for each task
        reps = []
    if args.elmo:
        log.info("Loading ELMo from files:")
        log.info("ELMO_OPT_PATH = %s", ELMO_OPT_PATH)
        log.info("ELMO_WEIGHTS_PATH = %s", ELMO_WEIGHTS_PATH)
        if args.elmo_chars_only:
            log.info("\tUsing ELMo character CNN only!")
            elmo_embedder = ElmoCharacterEncoder(options_file=ELMO_OPT_PATH,
                                                 weight_file=ELMO_WEIGHTS_PATH,
                                                 requires_grad=False)
            d_emb += 512
        else:
            log.info("\tUsing full ELMo! (separate scalars/task)")
            elmo_embedder = ElmoTokenEmbedderWrapper(
                options_file=ELMO_OPT_PATH,
                weight_file=ELMO_WEIGHTS_PATH,
                # Include pretrain task
                num_output_representations=len(reps) + 1,
                dropout=args.dropout)
            d_emb += 1024
        token_embedder["elmo"] = elmo_embedder

    embedder = ElmoTextFieldEmbedder(token_embedder, reps,
                                     elmo_chars_only=args.elmo_chars_only,
                                     sep_embs_for_skip=args.sep_embs_for_skip)

    assert d_emb, "You turned off all the embeddings, ya goof!"
    return d_emb, embedder, cove_emb


def build_module(task, model, d_sent, d_emb, vocab, embedder, args):
    ''' Build task-specific components for a task and add them to model. '''
    task_params = model._get_task_params(task.name)
    if isinstance(task, SingleClassificationTask):
        module = build_single_sentence_module(task, d_sent, task_params)
        setattr(model, '%s_mdl' % task.name, module)
    elif isinstance(task, (PairClassificationTask, PairRegressionTask,
                           PairOrdinalRegressionTask)):
        module = build_pair_sentence_module(task, d_sent, model, vocab,
                                            task_params)
        setattr(model, '%s_mdl' % task.name, module)
    elif isinstance(task, LanguageModelingTask):
        d_sent = args.d_hid + (args.skip_embs * d_emb)
        hid2voc = build_lm(task, d_sent, args)
        setattr(model, '%s_hid2voc' % task.name, hid2voc)
    elif isinstance(task, TaggingTask):
        hid2tag = build_tagger(task, d_sent, task.num_tags)
        setattr(model, '%s_mdl' % task.name, hid2tag)
    elif isinstance(task, EdgeProbingTask):
        module = edge_probing.EdgeClassifierModule(task, d_sent, task_params)
        setattr(model, '%s_mdl' % task.name, module)
    elif isinstance(task, Wiki103_Seq2Seq):
        attention = args.get("mt_attention", "bilinear")
        log.info("using {} attention".format(attention))
        decoder_params = Params({'input_dim': d_sent,
                                 'target_embedding_dim': 300,
                                 'max_decoding_steps': args.max_seq_len,
                                 'target_namespace': 'tokens',
                                 'attention': attention,
                                 'dropout': args.dropout,
                                 'scheduled_sampling_ratio': 0.0})
        decoder = Seq2SeqDecoder.from_params(vocab, decoder_params)
        setattr(model, '%s_decoder' % task.name, decoder)
    elif isinstance(task, (MTTask, Reddit_MTTask)):
        attention = args.get("mt_attention", "bilinear")
        log.info("using {} attention".format(attention))
        decoder_params = Params({'input_dim': d_sent,
                                 'target_embedding_dim': 300,
                                 'max_decoding_steps': 200,
                                 'target_namespace': 'targets',
                                 'attention': attention,
                                 'dropout': args.dropout,
                                 'scheduled_sampling_ratio': 0.0})
        decoder = Seq2SeqDecoder.from_params(vocab, decoder_params)
        setattr(model, '%s_decoder' % task.name, decoder)
    elif isinstance(task, SequenceGenerationTask):
        decoder, hid2voc = build_decoder(task, d_sent, vocab, embedder, args)
        setattr(model, '%s_decoder' % task.name, decoder)
        setattr(model, '%s_hid2voc' % task.name, hid2voc)

    elif isinstance(task, VAETask):
        decoder_params = Params({'input_dim': d_sent,
                                 'target_embedding_dim': 300,
                                 'max_decoding_steps': 200,
                                 'target_namespace': 'tokens',
                                 'attention': 'bilinear',
                                 'dropout': args.dropout,
                                 'scheduled_sampling_ratio': 0.0})
        decoder = Seq2SeqDecoder.from_params(vocab, decoder_params)
        setattr(model, '%s_decoder' % task.name, decoder)

    elif isinstance(task, GroundedTask):
        task.img_encoder = CNNEncoder(model_name='resnet', path=task.path)
    elif isinstance(task, RankingTask):
        pooler, dnn_ResponseModel = build_reddit_module(task, d_sent, task_params)
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
    _get_task_attr = lambda attr_name: config.get_task_attr(args, task_name,
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

    # For NLI probing tasks, might want to use a classifier trained on
    # something else (typically 'mnli').
    cls_task_name = _get_task_attr("use_classifier")
    params['use_classifier'] = cls_task_name or task_name  # default to this task

    return Params(params)


def build_reddit_module(task, d_inp, params):
    ''' Build a single classifier '''
    pooler = Pooler.from_params(d_inp, params['d_proj'])
    dnn_ResponseModel = nn.Sequential(nn.Linear(params['d_proj'], params['d_proj']),
                                        nn.Tanh(), nn.Linear(params['d_proj'], params['d_proj']),
                                        )
    #classifier = Classifier.from_params(params['d_proj'], task.n_classes, params)
    return pooler, dnn_ResponseModel


def build_single_sentence_module(task, d_inp, params):
    ''' Build a single classifier '''
    pooler = Pooler.from_params(d_inp, params['d_proj'])
    classifier = Classifier.from_params(params['d_proj'], task.n_classes, params)
    return SingleClassifier(pooler, classifier)


def build_pair_sentence_module(task, d_inp, model, vocab, params):
    ''' Build a pair classifier, shared if necessary '''

    def build_pair_attn(d_in, use_attn, d_hid_attn):
        ''' Build the pair model '''
        if not use_attn:
            pair_attn = None
        else:
            d_inp_model = 2 * d_in
            modeling_layer = s2s_e.by_name('lstm').from_params(
                Params({'input_size': d_inp_model, 'hidden_size': d_hid_attn,
                        'num_layers': 1, 'bidirectional': True}))
            pair_attn = AttnPairEncoder(vocab, modeling_layer,
                                        dropout=params["dropout"])
        return pair_attn

    if params["attn"]:
        pooler = Pooler.from_params(params["d_hid_attn"], params["d_hid_attn"], project=False)
        d_out = params["d_hid_attn"] * 2
    else:
        pooler = Pooler.from_params(d_inp, params["d_proj"], project=True)
        d_out = params["d_proj"]

    if params["shared_pair_attn"]:
        if not hasattr(model, "pair_attn"):
            pair_attn = build_pair_attn(d_inp, params["attn"], params["d_hid_attn"])
            model.pair_attn = pair_attn
        else:
            pair_attn = model.pair_attn
    else:
        pair_attn = build_pair_attn(d_inp, params["attn"], params["d_hid_attn"])

    n_classes = task.n_classes if hasattr(task, 'n_classes') else 1
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
                'hidden_size': args.d_hid_dec,
                'num_layers': args.n_layers_dec, 'bidirectional': False}))
    decoder = SentenceEncoder(vocab, embedder, 0, rnn)
    hid2voc = nn.Linear(args.d_hid_dec, args.max_word_v_size)
    return decoder, hid2voc


class MultiTaskModel(nn.Module):
    '''
    Giant model with task-specific components and a shared word and sentence encoder.
    '''

    def __init__(self, args, sent_encoder, vocab):
        ''' Args: sentence encoder '''
        super(MultiTaskModel, self).__init__()
        self.sent_encoder = sent_encoder
        self.combine_method = args.sent_combine_method
        self.vocab = vocab
        self.utilization = Average() if args.track_batch_utilization else None
        self.elmo = args.elmo and not args.elmo_chars_only
        self.sep_embs_for_skip = args.sep_embs_for_skip


    def forward(self, task, batch, predict=False):
        '''
        Pass inputs to correct forward pass

        Args:
            - task
            - batch

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
            out = self._pair_sentence_MNLI_diagnostic_forward(batch, task, predict)
        elif isinstance(task, (PairClassificationTask, PairRegressionTask,
                               PairOrdinalRegressionTask)):
            out = self._pair_sentence_forward(batch, task, predict)
        elif isinstance(task, LanguageModelingTask):
            out = self._lm_forward(batch, task, predict)
        elif isinstance(task, VAETask):
            out = self._vae_forward(batch, task, predict)
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
        elif isinstance(task, GroundedTask):
            out = self._grounded_classification_forward(batch, task, predict)
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

        if 'labels' in batch: # means we should compute loss
            if batch['labels'].dim() == 0:
                labels = batch['labels'].unsqueeze(0)
            elif batch['labels'].dim() == 1:
                labels = batch['labels']
            else:
                labels = batch['labels'].squeeze(-1)
            out['loss'] = F.cross_entropy(logits, labels)
            if isinstance(task, CoLATask):
                task.scorer2(logits, labels)
                _, preds = logits.max(dim=1)
                task.scorer1(labels, preds)
            else:
                task.scorer1(logits, labels)
                if task.scorer2 is not None:
                    task.scorer2(logits, labels)

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
        sent1, mask1 = self.sent_encoder(batch['input1'], task)
        sent2, mask2 = self.sent_encoder(batch['input2'], task)
        classifier = self._get_classifier(task)
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


    def _pair_sentence_forward(self, batch, task, predict):
        out = {}

        # embed the sentence
        sent1, mask1 = self.sent_encoder(batch['input1'], task)
        sent2, mask2 = self.sent_encoder(batch['input2'], task)
        classifier = self._get_classifier(task)
        logits = classifier(sent1, sent2, mask1, mask2)
        out['logits'] = logits
        out['n_exs'] = get_batch_size(batch)

        if 'labels' in batch:
            labels = batch['labels']
            labels = labels.squeeze(-1) if len(labels.size()) > 1 else labels
            if isinstance(task, JOCITask):
                logits = logits.squeeze(-1) if len(logits.size()) > 1 else logits
                out['loss'] = F.mse_loss(logits, labels)
                logits_np = logits.data.cpu().numpy()
                labels_np = labels.data.cpu().numpy()
                task.scorer1(mean_squared_error(logits_np, labels_np))
                task.scorer2(logits_np, labels_np)
            elif isinstance(task, STSBTask):
                logits = logits.squeeze(-1) if len(logits.size()) > 1 else logits
                out['loss'] = F.mse_loss(logits, labels)
                logits_np = logits.data.cpu().numpy()
                labels_np = labels.data.cpu().numpy()
                task.scorer1(logits_np, labels_np)
                task.scorer2(logits_np, labels_np)
            else:
                out['loss'] = F.cross_entropy(logits, labels)
                task.scorer1(logits, labels)
                if task.scorer2 is not None:
                    task.scorer2(logits, labels)

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
        ''' For caption and image ranking. This implementation is intended for Reddit'''
        out = {}
        # feed forwarding inputs through sentence encoders
        sent1, mask1 = self.sent_encoder(batch['input1'], task)
        sent2, mask2 = self.sent_encoder(batch['input2'], task)
        # pooler for both Input and Response
        sent_pooler = getattr(self, "%s_mdl" % task.name)
        sent_dnn = getattr(self, "%s_Response_mdl" % task.name) # dnn for Response
        sent1_rep = sent_pooler(sent1, mask1)
        sent2_rep_pool = sent_pooler(sent2, mask2)
        sent2_rep = sent_dnn(sent2_rep_pool)

        #  if 1:
        sent1_rep = sent1_rep/sent1_rep.norm(dim=1)[:, None]
        sent2_rep = sent2_rep/sent2_rep.norm(dim=1)[:, None]
        cos_simi = torch.mm(sent1_rep, sent2_rep.transpose(0,1))
        labels = torch.eye(len(cos_simi))

        # balancing pairs: #positive_pairs = batch_size, #negative_pairs = batch_size-1
        cos_simi_pos = torch.diag(cos_simi)
        cos_simi_neg = torch.diag(cos_simi, diagonal=1)
        cos_simi = torch.cat([cos_simi_pos, cos_simi_neg], dim=0)
        labels_pos = torch.diag(labels)
        labels_neg = torch.diag(labels, diagonal=1)
        labels = torch.cat([labels_pos, labels_neg], dim=0)
        labels = labels.cuda()
        #total_loss = torch.nn.BCEWithLogitsLoss(weight=weights)(cos_simi, labels)
        total_loss = torch.nn.BCEWithLogitsLoss()(cos_simi, labels)
        out['loss'] = total_loss

        pred = F.sigmoid(cos_simi).round()
        total_correct = torch.sum(pred == labels)
        batch_acc = total_correct.item()/len(labels)
        out["n_exs"] = len(labels)
        task.scorer1(batch_acc)

        return out


    def _vae_forward(self, batch, task, predict):
        ''' For translation, denoising, maybe language modeling? '''
        out = {}
        sent, sent_mask = self.sent_encoder(batch['inputs'], task)
        out['n_exs'] = get_batch_size(batch)

        if isinstance(task, VAETask):
            decoder = getattr(self, "%s_decoder" % task.name)
            out = decoder.forward(sent, sent_mask, batch['targs'])
            task.scorer1(math.exp(out['loss'].item()))
            return out
        if 'targs' in batch:
            pass

        if predict:
            pass

        return out

    def _seq_gen_forward(self, batch, task, predict):
        ''' For variational autoencoder '''
        out = {}
        sent, sent_mask = self.sent_encoder(batch['inputs'], task)
        out['n_exs'] = get_batch_size(batch)

        if isinstance(task, (MTTask, Reddit_MTTask)):
            decoder = getattr(self, "%s_decoder" % task.name)
            out.update(decoder.forward(sent, sent_mask, batch['targs']))
            task.scorer1(math.exp(out['loss'].item()))

            if not self.training and not isinstance(task, Wiki103_Seq2Seq):
                # bleu scoring
                bleu_score, unk_ratio_macroavg = beamsearch.generate_and_compute_bleu(decoder, sent, sent_mask, batch['targs']['words'], preds_file_path=task.preds_file_path, task=task)
                task.scorer2(bleu_score)
                task.scorer3(unk_ratio_macroavg)

            return out

        if 'targs' in batch:
            pass

        if predict:
            pass

        return out

    def _tagger_forward(self, batch, task, predict):
        ''' For language modeling? '''
        out = {}
        b_size, seq_len, _ = batch['inputs']['elmo'].size()
        seq_len -= 2
        sent_encoder = self.sent_encoder

        out['n_exs'] = get_batch_size(batch)
        if not isinstance(sent_encoder, BiLMEncoder):
            sent, mask = sent_encoder(batch['inputs'], task)
            sent = sent.masked_fill(1 - mask.byte(), 0)  # avoid NaNs
            sent = sent[:,1:-1,:]
            hid2tag = self._get_classifier(task)
            logits = hid2tag(sent)
            logits = logits.view(b_size * seq_len, -1)
            out['logits'] = logits
            targs = batch['targs']['words'][:,:seq_len].contiguous().view(-1)


        pad_idx = self.vocab.get_token_index(self.vocab._padding_token)
        out['loss'] = F.cross_entropy(logits, targs, ignore_index=pad_idx)
        task.scorer1(out['loss'].item())
        return out

    def _lm_forward(self, batch, task, predict):
        ''' For language modeling '''
        out = {}
        sent_encoder = self.sent_encoder
        assert_for_log(isinstance(sent_encoder._phrase_layer, BiLMEncoder),
                       "Not using LM for language modeling task!")
        assert_for_log('targs' in batch and 'words' in batch['targs'],
                       "Batch missing target words!")
        pad_idx = self.vocab.get_token_index(self.vocab._padding_token, 'tokens')
        b_size, seq_len = batch['targs']['words'].size()
        n_pad = batch['targs']['words'].eq(pad_idx).sum().item()
        out['n_exs'] = (b_size * seq_len - n_pad) * 2

        sent, mask = sent_encoder(batch['input'], task)
        sent = sent.masked_fill(1 - mask.byte(), 0)  # avoid NaNs

        # Split encoder outputs by direction
        split = int(self.sent_encoder._phrase_layer.get_output_dim() / 2)
        fwd, bwd = sent[:, :, :split], sent[:, :, split:split*2]
        if split * 2 < sent.size(2): # skip embeddings
           out_embs = sent[:, :, split*2:]
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
        assert logits.size(0) == targs.size(0), "Number of logits and targets differ!"
        out['loss'] = F.cross_entropy(logits, targs, ignore_index=pad_idx)
        task.scorer1(out['loss'].item())
        if predict:
            pass
        return out

    def _grounded_classification_forward(self, batch, task, predict):
        out = {}
        # embed the sentence, embed the image, map and classify
        sent_emb, sent_mask = self.sent_encoder(batch['input1'], task)
        batch_size = get_batch_size_from_field(batch['input1'])
        out['n_exs'] = batch_size

        ids = batch['ids'].cpu().squeeze(-1).data.numpy().tolist()
        labels = batch['labels'].cpu().squeeze(-1)
        labels = [int(item) for item in labels.data.numpy()]

        cos = task.metric_fn
        flags = Variable(torch.ones(batch_size))

        img_idx, preds, img_seq,sent_seq = 0, [], [], []
        for sent in sent_emb.split(1):
            seq_len = sent.size()[1]
            sent = task.pooler(sent.view(seq_len, -1))
            sent = torch.div(torch.sum(sent, dim=0), seq_len).cuda().reshape((1, -1))
            img_feat = torch.tensor(task.img_encoder.forward(int(ids[img_idx])), dtype=torch.float32).cuda()
            img_seq.append(img_feat); sent_seq.append(sent); img_idx += 1
            sim = cos(sent, img_feat).cuda()
            preds.append(sim.cpu().data.numpy()[0])

        img_emb = torch.stack(img_seq, dim=0); sent_emb = torch.stack(sent_seq, dim=0)
        metric = np.mean(preds)
        task.scorer1.__call__(metric)
        out['logits'] = torch.tensor(preds, dtype=torch.float32).reshape(1, -1)

        cos = task.loss_fn; flags = Variable(torch.ones(batch_size))
        out['loss'] = cos(
            torch.tensor(
                sent_emb.reshape(batch_size, -1), dtype=torch.float), torch.tensor(
                img_emb, dtype=torch.float).reshape(batch_size, -1), flags)

        if predict:
            out['preds'] = preds

        return out

    def get_elmo_mixing_weights(self, tasks=[]):
        ''' Get elmo mixing weights from text_field_embedder,
        since elmo should be in the same place every time.

        args:
            - text_field_embedder
            - mix_id: if we learned multiple mixing weights, which one we want
                to extract, usually 0

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
                    params[task.name] = get_elmo_mixing_weights(self.sent_encoder._text_field_embedder, task=task)
                else:
                    params["@pretrain@"] = get_elmo_mixing_weights(self.sent_encoder._text_field_embedder, task=None)
        return params
