'''AllenNLP models and functions for building them'''
import os
import sys
import ipdb as pdb
import logging as log
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.similarity_functions import LinearSimilarity, DotProductSimilarity
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
from allennlp.modules.elmo import Elmo

from tasks import STS14Task, STSBenchmarkTask, AcceptabilityTask
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

# CoVe stuff
if "cs.nyu.edu" in os.uname()[1] or "dgx" in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
else:
    PATH_PREFIX = '/beegfs/aw3272/'

PATH_TO_COVE = PATH_PREFIX + '/models/cove'
sys.path.append(PATH_TO_COVE)
from cove import MTLSTM as cove_lstm

# Elmo stuff
ELMO_OPT_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
ELMO_WEIGHTS_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long

logger = log.getLogger(__name__)  # pylint: disable=invalid-name

def build_model(args, vocab, pretrained_embs, tasks):
    '''Build model according to arguments

    args:
        - args (TODO): object with attributes:
        - vocab (Vocab):
        - pretrained_embs (TODO): word embeddings to use

    returns
    '''
    d_word, n_layers_highway = args.d_word, args.n_layers_highway

    # Build embedding layers
    if args.glove:
        word_embs = pretrained_embs
        train_embs = bool(args.train_words)
    else:
        log.info("\tLearning embeddings from scratch!")
        word_embs = None
        train_embs = True
    word_embedder = Embedding(vocab.get_vocab_size('tokens'), d_word, weight=word_embs,
                              trainable=train_embs,
                              padding_index=vocab.get_token_index('@@PADDING@@'))
    d_inp_phrase = 0

    # Handle elmo and cove
    token_embedder = {}
    if args.elmo:
        log.info("\tUsing ELMo embeddings!")
        if args.deep_elmo:
            n_reps = 2
            log.info("\tUsing deep ELMo embeddings!")
        else:
            n_reps = 1
        if args.elmo_no_glove:
            log.info("\tNOT using GLoVe embeddings!")
        else:
            token_embedder = {"words": word_embedder}
            log.info("\tUsing GLoVe embeddings!")
            d_inp_phrase += d_word
        elmo = Elmo(options_file=ELMO_OPT_PATH, weight_file=ELMO_WEIGHTS_PATH,
                    num_output_representations=n_reps)
        d_inp_phrase += 1024
    else:
        elmo = None
        token_embedder = {"words": word_embedder}
        d_inp_phrase += d_word
    text_field_embedder = BasicTextFieldEmbedder(token_embedder) if "words" in token_embedder \
                            else None
    d_hid_phrase = args.d_hid if args.pair_enc != 'bow' else d_inp_phrase

    if args.cove:
        cove_layer = cove_lstm(n_vocab=vocab.get_vocab_size('tokens'),
                               vectors=word_embedder.weight.data)
        d_inp_phrase += 600
        log.info("\tUsing CoVe embeddings!")
    else:
        cove_layer = None

    # Build encoders
    phrase_layer = s2s_e.by_name('lstm').from_params(Params({'input_size': d_inp_phrase,
                                                             'hidden_size': d_hid_phrase,
                                                             'num_layers': args.n_layers_enc,
                                                             'bidirectional': True}))
    if args.pair_enc == 'bow':
        sent_encoder = BoWSentEncoder(vocab, text_field_embedder) # maybe should take in CoVe/ELMO?
        pair_encoder = None # model will just run sent_encoder on both inputs
    else: # output will be 2 x d_hid_phrase (+ deep elmo)
        sent_encoder = HeadlessSentEncoder(vocab, text_field_embedder, n_layers_highway,
                                           phrase_layer, dropout=args.dropout,
                                           cove_layer=cove_layer, elmo_layer=elmo)
    d_single = 2 * d_hid_phrase + (args.elmo and args.deep_elmo) * 1024
    if args.pair_enc == 'simple': # output will be 4 x [2 x d_hid_phrase (+ deep elmo)]
        pair_encoder = HeadlessPairEncoder(vocab, text_field_embedder, n_layers_highway,
                                           phrase_layer, cove_layer=cove_layer, elmo_layer=elmo,
                                           dropout=args.dropout)
        d_pair = d_single
    elif args.pair_enc == 'attn':
        log.info("\tUsing attention!")
        d_inp_model = 4 * d_hid_phrase + (args.elmo and args.deep_elmo) * 1024
        d_hid_model = d_hid_phrase # make it as large as the original sentence encoding
        modeling_layer = s2s_e.by_name('lstm').from_params(Params({'input_size': d_inp_model,
                                                                   'hidden_size': d_hid_model,
                                                                   'num_layers':  1,
                                                                   'bidirectional': True}))
        pair_encoder = HeadlessPairAttnEncoder(vocab, text_field_embedder, n_layers_highway,
                                               phrase_layer, DotProductSimilarity(), modeling_layer,
                                               cove_layer=cove_layer, elmo_layer=elmo,
                                               deep_elmo=args.deep_elmo,
                                               dropout=args.dropout)
        d_pair = 2 * d_hid_phrase
        # output will be 4 x [2 x d_hid_model], where d_hid_model = 2 x d_hid_phrase
        #                = 4 x [2 x 2 x d_hid_phrase]

    # Build model and classifiers
    model = MultiTaskModel(args, sent_encoder, pair_encoder)
    build_classifiers(tasks, model, d_pair, d_single)
    if args.cuda >= 0:
        model = model.cuda()
    return model

def build_classifiers(tasks, model, d_pair, d_single):
    '''
    Build the classifier for each task
    '''
    for task in tasks:
        d_task =  d_pair * 4 if task.pair_input else d_single
        model.build_classifier(task, d_task)
    return


class MultiTaskModel(nn.Module):
    '''
    Playing around designing a class
    '''

    def __init__(self, args, sent_encoder, pair_encoder):
        '''

        Args:
        '''
        super(MultiTaskModel, self).__init__()
        self.sent_encoder = sent_encoder
        self.pair_encoder = pair_encoder
        self.pair_enc_type = args.pair_enc

        self.cls_type = args.classifier
        self.dropout_cls = args.classifier_dropout
        self.d_hid_cls = args.classifier_hid_dim

    def build_classifier(self, task, d_inp):
        ''' Build a task specific prediction layer and register it '''
        cls_type, dropout, d_hid = self.cls_type, self.dropout_cls, self.d_hid_cls
        if isinstance(task, (STSBenchmarkTask, STS14Task)) or cls_type == 'log_reg':
            layer = nn.Linear(d_inp, task.n_classes)
        elif cls_type == 'mlp':
            layer = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid), nn.Tanh(),
                                  nn.Dropout(p=dropout), nn.Linear(d_hid, task.n_classes))
        elif cls_type == 'fancy_mlp':
            layer = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid), nn.Tanh(),
                                  nn.Dropout(p=dropout), nn.Linear(d_hid, d_hid), nn.Tanh(),
                                  nn.Dropout(p=dropout), nn.Linear(d_hid, task.n_classes))
        else:
            raise ValueError("Unrecognized classifier!")

        setattr(self, '%s_pred_layer' % task.name, layer)

    def forward(self, task=None, input1=None, input2=None, label=None):
        '''
        Predict through model and task-specific prediction layer

        Args:
            - inputs (tuple(TODO))
            - pred_layer (nn.Module)
            - pair_input (int)

        Returns:
            - logits (TODO)
        '''
        pair_input = task.pair_input
        pred_layer = getattr(self, '%s_pred_layer' % task.name)
        if pair_input:
            if self.pair_enc_type == 'bow':
                sent1 = self.sent_encoder(input1)
                sent2 = self.sent_encoder(input2) # causes a bug with BiDAF
                logits = pred_layer(torch.cat([sent1, sent2, torch.abs(sent1 - sent2),
                                               sent1 * sent2], 1))
            else:
                pair_emb = self.pair_encoder(input1, input2)
                logits = pred_layer(pair_emb)

        else:
            sent_emb = self.sent_encoder(input1)
            logits = pred_layer(sent_emb)
        out = {'logits': logits}
        if label is not None:
            if isinstance(task, (STS14Task, STSBenchmarkTask)):
                loss = F.mse_loss(logits, label)
                label = label.squeeze(-1).data.cpu().numpy()
                logits = logits.squeeze(-1).data.cpu().numpy()
                task.scorer1(pearsonr(logits, label)[0])
                task.scorer2(spearmanr(logits, label)[0])
            elif isinstance(task, AcceptabilityTask):
                label = label.squeeze(-1)
                loss = F.cross_entropy(logits, label)
                task.scorer2(logits, label)
                label = label.data.cpu().numpy()
                _, preds = logits.max(dim=1)
                task.scorer1(matthews_corrcoef(label, preds.data.cpu().numpy()))
            else:
                label = label.squeeze(-1)
                loss = F.cross_entropy(logits, label)
                task.scorer1(logits, label)
                if task.scorer2 is not None:
                    task.scorer2(logits, label)
            out['loss'] = loss
        return out

class HeadlessPairEncoder(Model):
    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 cove_layer=None, elmo_layer=None, dropout=0.2, mask_lstms=True,
                 initializer=InitializerApplicator(), regularizer=None):
        super(HeadlessPairEncoder, self).__init__(vocab)#, regularizer)

        if text_field_embedder is None: # just using ELMo embeddings
            self._text_field_embedder = lambda x: x
            d_emb = 0
            self._highway_layer = lambda x: x
        else:
            self._text_field_embedder = text_field_embedder
            d_emb = text_field_embedder.get_output_dim()
            self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))
        self._phrase_layer = phrase_layer
        d_inp_phrase = phrase_layer.get_input_dim()
        self._cove = cove_layer
        self._elmo = elmo_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.output_dim = phrase_layer.get_output_dim()

        if (cove_layer is None and elmo_layer is None and d_emb != d_inp_phrase) \
            or (cove_layer is not None and d_emb + 600 != d_inp_phrase) \
            or (elmo_layer is not None and d_emb + 1024 != d_inp_phrase):
            raise ConfigurationError("The output dimension of the text_field_embedder "
                                     "(embedding_dim + char_cnn) must match the input "
                                     "dimension of the phrase_encoder. Found {} and {} "
                                     "respectively.".format(d_emb, d_inp_phrase))
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, s1, s2):
        # pylint: disable=arguments-differ
        """ """
        # Embeddings
        s1_embs = self._highway_layer(self._text_field_embedder(s1))
        s2_embs = self._highway_layer(self._text_field_embedder(s2))
        if self._elmo is not None:
            s1_elmo_embs = self._elmo(s1['elmo'])
            s2_elmo_embs = self._elmo(s2['elmo'])
            if "words" in s1:
                s1_embs = torch.cat([s1_embs, s1_elmo_embs['elmo_representations'][0]], dim=-1)
                s2_embs = torch.cat([s2_embs, s2_elmo_embs['elmo_representations'][0]], dim=-1)
            else:
                s1_embs = s1_elmo_embs['elmo_representations'][0]
                s2_embs = s2_elmo_embs['elmo_representations'][0]

        if self._cove is not None:
            s1_lens = torch.ne(s1['words'], self.pad_idx).long().sum(dim=-1).data
            s2_lens = torch.ne(s2['words'], self.pad_idx).long().sum(dim=-1).data
            s1_cove_embs = self._cove(s1['words'], s1_lens)
            s1_embs = torch.cat([s1_embs, s1_cove_embs], dim=-1)
            s2_cove_embs = self._cove(s2['words'], s2_lens)
            s2_embs = torch.cat([s2_embs, s2_cove_embs], dim=-1)
        s1_embs = self._dropout(s1_embs)
        s2_embs = self._dropout(s2_embs)

        # Set up masks
        s1_mask = util.get_text_field_mask(s1)
        s2_mask = util.get_text_field_mask(s2)
        s1_lstm_mask = s1_mask.float() if self._mask_lstms else None
        s2_lstm_mask = s2_mask.float() if self._mask_lstms else None

        # Sentence encodings with LSTMs
        s1_enc = self._phrase_layer(s1_embs, s1_lstm_mask)
        s2_enc = self._phrase_layer(s2_embs, s2_lstm_mask)
        if self._elmo is not None and len(s1_elmo_embs['elmo_representations']) > 1:
            s1_enc = torch.cat([s1_enc, s1_elmo_embs['elmo_representations'][1]], dim=-1)
            s2_enc = torch.cat([s2_enc, s2_elmo_embs['elmo_representations'][1]], dim=-1)
        s1_enc = self._dropout(s1_enc)
        s2_enc = self._dropout(s2_enc)

        # Max pooling
        s1_mask = s1_mask.unsqueeze(dim=-1)
        s2_mask = s2_mask.unsqueeze(dim=-1)
        s1_enc.data.masked_fill_(1 - s1_mask.byte().data, -float('inf'))
        s2_enc.data.masked_fill_(1 - s2_mask.byte().data, -float('inf'))
        s1_enc, _ = s1_enc.max(dim=1)
        s2_enc, _ = s2_enc.max(dim=1)

        return torch.cat([s1_enc, s2_enc, torch.abs(s1_enc - s2_enc), s1_enc * s2_enc], 1)

class BoWSentEncoder(Model):
    def __init__(self, vocab, text_field_embedder, initializer=InitializerApplicator(),
                 regularizer=None):
        super(BoWSentEncoder, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self.output_dim = text_field_embedder.get_output_dim()
        initializer(self)

    def forward(self, question):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.

        Returns
        -------
        pair_rep : torch.FloatTensor?
            Tensor representing the final output of the BiDAF model
            to be plugged into the next module

        """
        word_char_embs = self._text_field_embedder(question)
        question_mask = util.get_text_field_mask(question).float()
        return word_char_embs.mean(1) # need to get # nonzero elts


class HeadlessSentEncoder(Model):
    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 cove_layer=None, elmo_layer=None, dropout=0.2, mask_lstms=True,
                 initializer=InitializerApplicator(), regularizer= None):
        super(HeadlessSentEncoder, self).__init__(vocab)#, regularizer)

        if text_field_embedder is None:
            self._text_field_embedder = lambda x: x
            d_emb = 0
            self._highway_layer = lambda x: x
        else:
            self._text_field_embedder = text_field_embedder
            d_emb = text_field_embedder.get_output_dim()
            self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))
        self._phrase_layer = phrase_layer
        d_inp_phrase = phrase_layer.get_input_dim()
        self._cove = cove_layer
        self._elmo = elmo_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.output_dim = phrase_layer.get_output_dim()

        #if d_emb != d_inp_phrase:
        if (cove_layer is None and elmo_layer is None and d_emb != d_inp_phrase) \
                or (cove_layer is not None and d_emb + 600 != d_inp_phrase) \
                or (elmo_layer is not None and d_emb + 1024 != d_inp_phrase):
            raise ConfigurationError("The output dimension of the text_field_embedder "
                                     "must match the input dimension of "
                                     "the phrase_encoder. Found {} and {} respectively." \
                                     .format(d_emb, d_inp_phrase))
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, sent):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sent : Dict[str, torch.LongTensor]
            From a ``TextField``.

        Returns
        -------
        """
        sent_embs = self._highway_layer(self._text_field_embedder(sent))
        if self._cove is not None:
            sent_lens = torch.ne(sent['words'], self.pad_idx).long().sum(dim=-1).data
            sent_cove_embs = self._cove(sent['words'], sent_lens)
            sent_embs = torch.cat([sent_embs, sent_cove_embs], dim=-1)
        if self._elmo is not None:
            elmo_embs = self._elmo(sent['elmo'])
            if "words" in sent:
                sent_embs = torch.cat([sent_embs, elmo_embs['elmo_representations'][0]], dim=-1)
            else:
                sent_embs = elmo_embs['elmo_representations'][0]
        sent_embs = self._dropout(sent_embs)

        sent_mask = util.get_text_field_mask(sent).float()
        sent_lstm_mask = sent_mask if self._mask_lstms else None

        sent_enc = self._phrase_layer(sent_embs, sent_lstm_mask)
        if self._elmo is not None and len(elmo_embs['elmo_representations']) > 1:
            sent_enc = torch.cat([sent_enc, elmo_embs['elmo_representations'][1]], dim=-1)
        sent_enc = self._dropout(sent_enc)

        sent_mask = sent_mask.unsqueeze(dim=-1)
        sent_enc.data.masked_fill_(1 - sent_mask.byte().data, -float('inf'))
        return sent_enc.max(dim=1)[0]

class HeadlessPairAttnEncoder(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).

    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    attention_similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in after the bidirectional
        attention.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 attention_similarity_function, modeling_layer,
                 cove_layer=None, elmo_layer=None, deep_elmo=False,
                 dropout=0.2, mask_lstms=True,
                 initializer=InitializerApplicator(), regularizer=None):
        super(HeadlessPairAttnEncoder, self).__init__(vocab)#, regularizer)

        if text_field_embedder is None: # just using ELMo embeddings
            self._text_field_embedder = lambda x: x
            d_emb = 0
            self._highway_layer = lambda x: x
        else:
            self._text_field_embedder = text_field_embedder
            d_emb = text_field_embedder.get_output_dim()
            self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))

        self._phrase_layer = phrase_layer
        self._matrix_attention = MatrixAttention(attention_similarity_function)
        self._modeling_layer = modeling_layer
        self._cove = cove_layer
        self._elmo = elmo_layer
        self._deep_elmo = deep_elmo
        self.pad_idx = vocab.get_token_index(vocab._padding_token)

        d_inp_phrase = phrase_layer.get_input_dim()
        d_out_phrase = phrase_layer.get_output_dim()
        d_out_model = modeling_layer.get_output_dim()
        d_inp_model = modeling_layer.get_input_dim()
        self.output_dim = d_out_model

        if (elmo_layer is None and d_inp_model != 2 * d_out_phrase) or \
            (elmo_layer is not None and not deep_elmo and d_inp_model != 2 * d_out_phrase) or \
            (elmo_layer is not None and deep_elmo and d_inp_model != 2 * d_out_phrase + 1024):
            raise ConfigurationError("The input dimension to the modeling_layer must be "
                                     "equal to 4 times the encoding dimension of the phrase_layer. "
                                     "Found {} and 4 * {} respectively.".format(d_inp_model, d_out_phrase))
        if (cove_layer is None and elmo_layer is None and d_emb != d_inp_phrase) \
            or (cove_layer is not None and d_emb + 600 != d_inp_phrase) \
            or (elmo_layer is not None and d_emb + 1024 != d_inp_phrase):
            raise ConfigurationError("The output dimension of the text_field_embedder "
                                     "(embedding_dim + char_cnn) must match the input "
                                     "dimension of the phrase_encoder. Found {} and {} "
                                     "respectively.".format(d_emb, d_inp_phrase))
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, s1, s2):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        s1 : Dict[str, torch.LongTensor]
            From a ``TextField``.
        s2 : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this s2 contains the answer to the
            s1, and predicts the beginning and ending positions of the answer within the
            s2.

        Returns
        -------
        pair_rep : torch.FloatTensor?
            Tensor representing the final output of the BiDAF model
            to be plugged into the next module

        """
        s1_embs = self._highway_layer(self._text_field_embedder(s1))
        s2_embs = self._highway_layer(self._text_field_embedder(s2))
        if self._elmo is not None:
            s1_elmo_embs = self._elmo(s1['elmo'])
            s2_elmo_embs = self._elmo(s2['elmo'])
            if "words" in s1:
                s1_embs = torch.cat([s1_embs, s1_elmo_embs['elmo_representations'][0]], dim=-1)
                s2_embs = torch.cat([s2_embs, s2_elmo_embs['elmo_representations'][0]], dim=-1)
            else:
                s1_embs = s1_elmo_embs['elmo_representations'][0]
                s2_embs = s2_elmo_embs['elmo_representations'][0]
        if self._cove is not None:
            s1_lens = torch.ne(s1['words'], self.pad_idx).long().sum(dim=-1).data
            s2_lens = torch.ne(s2['words'], self.pad_idx).long().sum(dim=-1).data
            s1_cove_embs = self._cove(s1['words'], s1_lens)
            s1_embs = torch.cat([s1_embs, s1_cove_embs], dim=-1)
            s2_cove_embs = self._cove(s2['words'], s2_lens)
            s2_embs = torch.cat([s2_embs, s2_cove_embs], dim=-1)
        s1_embs = self._dropout(s1_embs)
        s2_embs = self._dropout(s2_embs)

        if self._mask_lstms:
            s1_mask = s1_lstm_mask = util.get_text_field_mask(s1).float()
            s2_mask = s2_lstm_mask = util.get_text_field_mask(s2).float()
            s1_mask_2 = util.get_text_field_mask(s1).float()
            s2_mask_2 = util.get_text_field_mask(s2).float()
        else:
            s1_lstm_mask, s2_lstm_mask, s2_lstm_mask_2 = None, None, None

        s1_enc = self._phrase_layer(s1_embs, s1_lstm_mask)
        s2_enc = self._phrase_layer(s2_embs, s2_lstm_mask)

        # Similarity matrix
        # Shape: (batch_size, s2_length, s1_length)
        similarity_mat = self._matrix_attention(s2_enc, s1_enc)

        # s2 representation
        # Shape: (batch_size, s2_length, s1_length)
        s2_s1_attention = util.last_dim_softmax(similarity_mat, s1_mask)
        # Shape: (batch_size, s2_length, encoding_dim)
        s2_s1_vectors = util.weighted_sum(s1_enc, s2_s1_attention)
        # batch_size, seq_len, 4*enc_dim
        s2_w_context = torch.cat([s2_enc, s2_s1_vectors], 2)
        # s1 representation, using same attn method as for the s2 representation
        s1_s2_attention = util.last_dim_softmax(similarity_mat.transpose(1, 2).contiguous(), s2_mask)
        # Shape: (batch_size, s1_length, encoding_dim)
        s1_s2_vectors = util.weighted_sum(s2_enc, s1_s2_attention)
        s1_w_context = torch.cat([s1_enc, s1_s2_vectors], 2)
        if self._elmo is not None and self._deep_elmo:
            s1_w_context = torch.cat([s1_w_context, s1_elmo_embs['elmo_representations'][1]], dim=-1)
            s2_w_context = torch.cat([s2_w_context, s2_elmo_embs['elmo_representations'][1]], dim=-1)
        s1_w_context = self._dropout(s1_w_context)
        s2_w_context = self._dropout(s2_w_context)

        modeled_s2 = self._dropout(self._modeling_layer(s2_w_context, s2_lstm_mask))
        s2_mask_2 = s2_mask_2.unsqueeze(dim=-1)
        modeled_s2.data.masked_fill_(1 - s2_mask_2.byte().data, -float('inf'))
        s2_enc_attn = modeled_s2.max(dim=1)[0]
        modeled_s1 = self._dropout(self._modeling_layer(s1_w_context, s1_lstm_mask))
        s1_mask_2 = s1_mask_2.unsqueeze(dim=-1)
        modeled_s1.data.masked_fill_(1 - s1_mask_2.byte().data, -float('inf'))
        s1_enc_attn = modeled_s1.max(dim=1)[0]

        return torch.cat([s1_enc_attn, s2_enc_attn, torch.abs(s1_enc_attn - s2_enc_attn),
                          s1_enc_attn * s2_enc_attn], 1)


    @classmethod
    def from_params(cls, vocab, params):
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        num_highway_layers = params.pop("num_highway_layers")
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
        dropout = params.pop('dropout', 0.2)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        mask_lstms = params.pop('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab, text_field_embedder=text_field_embedder,
                   num_highway_layers=num_highway_layers, phrase_layer=phrase_layer,
                   attention_similarity_function=similarity_function, modeling_layer=modeling_layer,
                   dropout=dropout, mask_lstms=mask_lstms,
                   initializer=initializer, regularizer=regularizer)
