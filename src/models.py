'''AllenNLP models and functions for building them'''
import os
import sys
import logging as log
import ipdb as pdb # pylint: disable=unused-import

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.models.model import Model
from allennlp.modules import Highway, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.similarity_functions import DotProductSimilarity
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
from allennlp.modules.elmo import Elmo

from tasks import STSBTask, CoLATask, SSTTask, \
                  PairClassificationTask, SingleClassificationTask, \
                  PairRegressionTask, RankingTask, \
                  SequenceGenerationTask, LanguageModelingTask
from modules import RNNEncoder, BoWSentEncoder, \
                    AttnPairEncoder, SimplePairEncoder

logger = log.getLogger(__name__)  # pylint: disable=invalid-name

if "cs.nyu.edu" in os.uname()[1] or "dgx" in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
else:
    PATH_PREFIX = '/beegfs/aw3272/'

# CoVe stuff
PATH_TO_COVE = PATH_PREFIX + '/models/cove'
sys.path.append(PATH_TO_COVE)
try:
    from cove import MTLSTM as cove_lstm
except ImportError:
    logger.info("Failed to import CoVE!")


# Elmo stuff
ELMO_OPT_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
ELMO_WEIGHTS_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long


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
    if args.sent_enc == 'bow':
        sent_encoder = BoWSentEncoder(vocab, text_field_embedder)
    elif args.sent_enc == 'rnn': # output will be 2 x d_hid_phrase (+ deep elmo)
        phrase_layer = s2s_e.by_name('lstm').from_params(
            Params({'input_size': d_inp_phrase, 'hidden_size': d_hid_phrase,
                    'num_layers': args.n_layers_enc, 'bidirectional': True}))

        sent_encoder = RNNEncoder(vocab, text_field_embedder, n_layers_highway,
                                  phrase_layer, dropout=args.dropout,
                                  cove_layer=cove_layer, elmo_layer=elmo)

    d_single = 2 * d_hid_phrase + (args.elmo and args.deep_elmo) * 1024
    if args.pair_enc == 'simple': # output will be 4 x [2 x d_hid_phrase (+ deep elmo)]
        pair_encoder = SimplePairEncoder(vocab)
        d_pair = d_single
    elif args.pair_enc == 'attn':
        log.info("\tUsing attention!")
        d_inp_model = 4 * d_hid_phrase + (args.elmo and args.deep_elmo) * 1024
        d_hid_model = d_hid_phrase # make it as large as the original sentence encoding
        modeling_layer = s2s_e.by_name('lstm').from_params(Params({'input_size': d_inp_model,
                                                                   'hidden_size': d_hid_model,
                                                                   'num_layers':  1,
                                                                   'bidirectional': True}))
        pair_encoder = AttnPairEncoder(vocab, DotProductSimilarity(),
                                       modeling_layer, dropout=args.dropout)

        d_pair = 2 * d_hid_phrase
        # output will be 4 x [2 x d_hid_model], where d_hid_model = 2 x d_hid_phrase
        #                = 4 x [2 x 2 x d_hid_phrase]

    # Build model and classifiers
    model = MultiTaskModel(args, sent_encoder, pair_encoder)
    build_modules(tasks, model, d_pair, d_single, vocab, text_field_embedder, args)
    if args.cuda >= 0:
        model = model.cuda()
    return model

def build_modules(tasks, model, d_pair, d_single, vocab, embedder, args):
    ''' Build task-specific components for each task and add them to model '''
    for task in tasks:
        if isinstance(task, (SingleClassificationTask, PairClassificationTask)):
            d_task = d_pair * 4 if isinstance(task, PairClassificationTask) else d_single
            module = build_classifier(task, d_task, args)
            setattr(model, '%s_mdl' % task.name, module)
        elif isinstance(task, PairRegressionTask):
            module = build_regressor(task, d_pair * 4, args)
            setattr(model, '%s_mdl' % task.name, module)
        elif isinstance(task, LanguageModelingTask):
            hid2voc = build_lm(task, d_pair, args)
            setattr(model, '%s_hid2voc' % task.name, hid2voc)
        elif isinstance(task, SequenceGenerationTask):
            decoder, hid2voc = build_decoder(task, d_pair, vocab, embedder, args)
            setattr(model, '%s_decoder' % task.name, decoder)
            setattr(model, '%s_hid2voc' % task.name, hid2voc)
        elif isinstance(task, RankingTask):
            pass
        else:
            raise ValueError("Module not found for %s", task.name)
    return

def build_classifier(task, d_inp, args):
    ''' Build a task specific classifier '''
    cls_type, dropout, d_hid = \
            args.classifier, args.classifier_dropout, args.classifier_hid_dim
    if isinstance(task, STSBTask) or cls_type == 'log_reg':
        classifier = nn.Linear(d_inp, task.n_classes)
    elif cls_type == 'mlp':
        classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid),
                                   nn.Tanh(), nn.Dropout(p=dropout),
                                   nn.Linear(d_hid, task.n_classes))
    elif cls_type == 'fancy_mlp':
        classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid),
                                   nn.Tanh(), nn.Dropout(p=dropout),
                                   nn.Linear(d_hid, d_hid), nn.Tanh(),
                                   nn.Dropout(p=dropout), nn.Linear(d_hid, task.n_classes))
    else:
        raise ValueError("Unrecognized classifier!")

    return classifier

def build_regressor(task, d_inp, args):
    ''' Build a task specific regressor '''
    cls_type, dropout, d_hid = \
            args.classifier, args.classifier_dropout, args.classifier_hid_dim
    if isinstance(task, STSBTask) or cls_type == 'log_reg':
        regressor = nn.Linear(d_inp, 1)
    elif cls_type == 'mlp':
        regressor = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid),
                                  nn.Tanh(), nn.Dropout(p=dropout),
                                  nn.Linear(d_hid, 1))
    elif cls_type == 'fancy_mlp':
        regressor = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid),
                                  nn.Tanh(), nn.Dropout(p=dropout),
                                  nn.Linear(d_hid, d_hid), nn.Tanh(),
                                  nn.Dropout(p=dropout), nn.Linear(d_hid, 1))
    return regressor

def build_lm(task, d_inp, args):
    hid2voc = nn.Linear(d_inp, args.max_word_v_size)
    return hid2voc

def build_decoder(task, d_inp, vocab, embedder, args):
    ''' Build a task specific decoder

    TODO: handle different vocabs (languages)?
    '''
    rnn = s2s_e.by_name('lstm').from_params(
        Params({'input_size': embedder.get_output_dim(),
                'hidden_size': args.d_hid_dec,
                'num_layers': args.n_layers_dec, 'bidirectional': False}))
    decoder = RNNEncoder(vocab, embedder, 0, rnn)
    hid2voc = nn.Linear(args.d_hid_dec, args.max_word_v_size)
    return decoder, hid2voc

class MultiTaskModel(nn.Module):
    '''
    Giant model with task-specific components and a shared word and sentence encoder.
    '''

    def __init__(self, args, sent_encoder, pair_encoder):
        '''

        Args:
        '''
        super(MultiTaskModel, self).__init__()
        self.sent_encoder = sent_encoder
        self.pair_encoder = pair_encoder
        self.pair_enc_type = args.pair_enc

    def forward(self, task, batch):
        '''
        Pass inputs to correct forward pass

        Args:
            - task
            - batch

        Returns:
            - out: dictionary containing task outputs and loss if label was in batch
        '''

        if isinstance(task, SingleClassificationTask):
            out = self._single_classification_forward(batch, task)
        elif isinstance(task, PairClassificationTask):
            out = self._pair_classification_forward(batch, task)
        elif isinstance(task, PairRegressionTask):
            out = self._pair_regression_forward(batch, task)
        elif isinstance(task, SequenceGenerationTask):
            out = self._seq_gen_forward(batch, task)
        elif isinstance(task, RankingTask):
            out = self._ranking_forward(batch, task)

        else:
            raise ValueError("Task-specific components not found!")
        return out

    def _single_classification_forward(self, batch, task):
        out = {}

        # embed the sentence
        sent_embs, sent_mask = self.sent_encoder(batch['input1'])
        sent_emb = sent_embs.max(dim=1)[0]

        # pass to a task specific classifier
        classifier = getattr(self, "%s_mdl" % task.name)
        logits = classifier(sent_emb)

        if 'labels' in batch:
            labels = batch['labels'].squeeze(-1)
            out['loss'] = F.cross_entropy(logits, labels)
            if isinstance(task, CoLATask):
                task.scorer2(logits, labels)
                labels = labels.data.cpu().numpy()
                _, preds = logits.max(dim=1)
                task.scorer1(matthews_corrcoef(labels, preds.data.cpu().numpy()))
            else:
                task.scorer1(logits, labels)
                if task.scorer2 is not None:
                    task.scorer2(logits, labels)
        out['logits'] = logits
        return out

    def _pair_classification_forward(self, batch, task):
        out = {}

        # embed the sentence
        s1, s1_mask = self.sent_encoder(batch['input1'])
        s2, s2_mask = self.sent_encoder(batch['input2'])
        pair_emb = self.pair_encoder(s1, s2, s1_mask, s2_mask)

        # pass to a task specific classifier
        classifier = getattr(self, "%s_mdl" % task.name)
        logits = classifier(pair_emb) # might want to pass sent_embs

        if 'labels' in batch:
            labels = batch['labels'].squeeze(-1)
            task.scorer1(logits, labels)
            if task.scorer2 is not None:
                task.scorer2(logits, labels)
            out['loss'] = F.cross_entropy(logits, labels)
        out['logits'] = logits
        return out

    def _pair_regression_forward(self, batch, task):
        ''' For STS-B '''
        out = {}
        s1, s1_mask = self.sent_encoder(batch['input1'])
        s2, s2_mask = self.sent_encoder(batch['input2'])
        pair_emb = self.pair_encoder(s1, s2, s1_mask, s2_mask)

        # pass to a task specific classifier
        regressor = getattr(self, "%s_mdl" % task.name)
        scores = regressor(pair_emb) # might want to pass sent_embs

        out['logits'] = scores # maybe change the name here?
        if 'labels' in batch:
            labels = batch['labels']
            out['loss'] = F.mse_loss(scores, labels)
            if isinstance(task, STSBTask):
                scores = scores.squeeze(-1).data.cpu().numpy()
                labels = labels.squeeze(-1).data.cpu().numpy()
                task.scorer1(pearsonr(scores, labels)[0])
                task.scorer2(spearmanr(scores, labels)[0])
        return out

    def _seq_gen_forward(self, batch, task):
        ''' For translation, denoising, maybe language modeling? '''
        out = {}
        b_size, seq_len = batch['inputs']['words'].size()
        sent, sent_mask = self.sent_encoder(batch['inputs'])

        if isinstance(task, LanguageModelingTask):
            hid2voc = getattr(self, "%s_hid2voc" % task.name)
            logits = hid2voc(sent)
            logits = logits.view(b_size * seq_len, -1)
        else:
            pass
        out['logits'] = logits

        if 'targs' in batch:
            targs = batch['targs']['words'].view(-1)
            out['loss'] = F.cross_entropy(logits, targs, ignore_index=0) # some pad index
            task.scorer1(out['loss'].item())
        return out

    def _ranking_forward(self, batch, task):
        ''' For caption and image ranking '''
        raise NotImplementedError
