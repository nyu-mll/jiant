'''Core model and functions for building it.

If you are adding a new task, you should [...]'''
import sys
import logging as log
import ipdb as pdb  # pylint: disable=unused-import

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, mean_squared_error

from allennlp.common import Params
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed
from allennlp.nn import util
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder, \
    ElmoTokenEmbedder
from allennlp.modules.similarity_functions import DotProductSimilarity
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder

from tasks import STSBTask, CoLATask, SSTTask, \
    PairClassificationTask, SingleClassificationTask, \
    PairRegressionTask, RankingTask, \
    SequenceGenerationTask, LanguageModelingTask, \
    PairOrdinalRegressionTask, JOCITask
from modules import SentenceEncoder, BoWSentEncoder, \
    AttnPairEncoder, SimplePairEncoder, MaskedStackedSelfAttentionEncoder, \
    BiLMEncoder, ElmoCharacterEncoder
from utils import combine_hidden_states

# Elmo stuff
ELMO_OPT_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"  # pylint: disable=line-too-long
ELMO_WEIGHTS_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"  # pylint: disable=line-too-long


def build_model(args, vocab, pretrained_embs, tasks):
    '''Build model according to args '''

    # Build embeddings.
    d_emb, embedder, cove_emb = build_embeddings(args, vocab, pretrained_embs)
    d_sent = args.d_hid

    # Build single sentence encoder: the main component of interest
    # Need special handling for language modeling
    if sum([isinstance(task, LanguageModelingTask) for task in tasks]):
        if args.bidirectional:
            if args.sent_enc == 'rnn':
                fwd = s2s_e.by_name('lstm').from_params(
                    Params({'input_size': d_emb, 'hidden_size': args.d_hid,
                            'num_layers': args.n_layers_enc, 'bidirectional': False}))
                bwd = s2s_e.by_name('lstm').from_params(
                    Params({'input_size': d_emb, 'hidden_size': args.d_hid,
                            'num_layers': args.n_layers_enc, 'bidirectional': False}))
            elif args.sent_enc == 'transformer':
                fwd = MaskedStackedSelfAttentionEncoder(input_dim=d_emb,
                                                        hidden_dim=args.d_hid,
                                                        projection_dim=args.d_proj,
                                                        feedforward_hidden_dim=args.d_ff,
                                                        num_layers=args.n_layers_enc,
                                                        num_attention_heads=args.n_heads)
                bwd = MaskedStackedSelfAttentionEncoder(input_dim=d_emb,
                                                        hidden_dim=args.d_hid,
                                                        projection_dim=args.d_proj,
                                                        feedforward_hidden_dim=args.d_ff,
                                                        num_layers=args.n_layers_enc,
                                                        num_attention_heads=args.n_heads)
            sent_encoder = BiLMEncoder(vocab, embedder, args.n_layers_highway,
                                       fwd, bwd, dropout=args.dropout,
                                       cove_layer=cove_emb)
        else:  # not bidirectional
            if args.sent_enc == 'rnn':
                fwd = s2s_e.by_name('lstm').from_params(
                    Params({'input_size': d_emb, 'hidden_size': args.d_hid,
                            'num_layers': args.n_layers_enc, 'bidirectional': False}))
            elif args.sent_enc == 'transformer':
                fwd = MaskedStackedSelfAttentionEncoder(input_dim=d_emb,
                                                        hidden_dim=args.d_hid,
                                                        projection_dim=args.d_proj,
                                                        feedforward_hidden_dim=args.d_ff,
                                                        num_layers=args.n_layers_enc,
                                                        num_attention_heads=args.n_heads)
            sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                           fwd, dropout=args.dropout,
                                           cove_layer=cove_emb)
    elif args.sent_enc == 'bow':
        sent_encoder = BoWSentEncoder(vocab, embedder)
        d_sent = d_emb
    elif args.sent_enc == 'rnn':
        sent_rnn = s2s_e.by_name('lstm').from_params(
            Params({'input_size': d_emb, 'hidden_size': args.d_hid,
                    'num_layers': args.n_layers_enc,
                    'bidirectional': args.bidirectional}))
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       sent_rnn, dropout=args.dropout,
                                       cove_layer=cove_emb)
        d_sent = (1 + args.bidirectional) * args.d_hid
    elif args.sent_enc == 'transformer':
        transformer = StackedSelfAttentionEncoder(input_dim=d_emb,
                                                  hidden_dim=args.d_hid,
                                                  projection_dim=args.d_proj,
                                                  feedforward_hidden_dim=args.d_ff,
                                                  num_layers=args.n_layers_enc,
                                                  num_attention_heads=args.n_heads)
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       transformer, dropout=args.dropout,
                                       cove_layer=cove_emb)

    # Build model and classifiers
    model = MultiTaskModel(args, sent_encoder, vocab)
    build_modules(tasks, model, d_sent, vocab, embedder, args)
    if args.cuda >= 0:
        model = model.cuda()
    log.info(model)
    return model


def build_embeddings(args, vocab, pretrained_embs=None):
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
    if args.elmo:
        if args.elmo_chars_only:
            log.info("\tUsing ELMo character CNN only!")
            #elmo_embedder = elmo_embedder._elmo._elmo_lstm._token_embedder
            elmo_embedder = ElmoCharacterEncoder(options_file=ELMO_OPT_PATH,
                                                 weight_file=ELMO_WEIGHTS_PATH,
                                                 requires_grad=False)
            d_emb += 512
        else:
            log.info("\tUsing full ELMo!")
            elmo_embedder = ElmoTokenEmbedder(options_file=ELMO_OPT_PATH,
                                              weight_file=ELMO_WEIGHTS_PATH,
                                              dropout=args.dropout)
            d_emb += 1024

        token_embedder["elmo"] = elmo_embedder

    embedder = BasicTextFieldEmbedder(token_embedder)
    assert d_emb, "You turned off all the embeddings, ya goof!"
    return d_emb, embedder, cove_emb


def build_modules(tasks, model, d_sent, vocab, embedder, args):
    ''' Build task-specific components for each task and add them to model '''
    for task in tasks:
        if isinstance(task, SingleClassificationTask):
            module = build_classifier(task, d_sent, args)
            setattr(model, '%s_mdl' % task.name, module)
        elif isinstance(task, PairClassificationTask):
            module = build_pair_classifier(task, d_sent, model, vocab, args)
            setattr(model, '%s_mdl' % task.name, module)
        elif isinstance(task, PairRegressionTask):
            module = build_regressor(task, d_sent * 4, args)
            setattr(model, '%s_mdl' % task.name, module)
        elif isinstance(task, PairOrdinalRegressionTask):
            regressor = build_regressor(task, d_sent * 4, args, model, vocab)
            setattr(model, '%s_mdl' % task.name, regressor)
        elif isinstance(task, LanguageModelingTask):
            hid2voc = build_lm(task, d_sent, args)
            setattr(model, '%s_hid2voc' % task.name, hid2voc)
        elif isinstance(task, SequenceGenerationTask):
            decoder, hid2voc = build_decoder(task, d_sent, vocab, embedder, args)
            setattr(model, '%s_decoder' % task.name, decoder)
            setattr(model, '%s_hid2voc' % task.name, hid2voc)
        elif isinstance(task, RankingTask):
            pass
        else:
            raise ValueError("Module not found for %s" % task.name)
    return


def build_classifier(task, d_inp, args):
    ''' Build a task specific classifier '''
    cls_type, dropout, d_hid = \
        args.classifier, args.classifier_dropout, args.classifier_hid_dim
    if isinstance(task, STSBTask) or cls_type == 'log_reg':
        classifier = nn.Linear(d_inp, task.n_classes)
    elif cls_type == 'mlp':
        classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid),
                                   nn.Tanh(), nn.LayerNorm(d_hid), nn.Dropout(p=dropout),
                                   nn.Linear(d_hid, task.n_classes))
    elif cls_type == 'fancy_mlp':  # what they did in InferSent
        classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid),
                                   nn.Tanh(), nn.LayerNorm(d_hid), nn.Dropout(p=dropout),
                                   nn.Linear(d_hid, d_hid), nn.Tanh(), nn.LayerNorm(d_hid),
                                   nn.Dropout(p=dropout), nn.Linear(d_hid, task.n_classes))
    else:
        raise ValueError("Classifier type not found!")

    return classifier


def build_pair_classifier(task, d_inp, model, vocab, args):
    ''' Build a pair classifier, shared if necessary '''

    def build_pair_encoder():
        if args.pair_enc == 'simple':
            pair_encoder = SimplePairEncoder(vocab, args.sent_combine_method)
            d_inp_classifier = 4 * d_inp
        elif args.pair_enc == 'attn':
            d_inp_model = 2 * d_inp
            d_hid_model = d_inp  # make it as large as the original sentence emb
            modeling_layer = s2s_e.by_name('lstm').from_params(
                Params({'input_size': d_inp_model, 'hidden_size': d_hid_model,
                        'num_layers': 1, 'bidirectional': True}))
            pair_encoder = AttnPairEncoder(vocab, DotProductSimilarity(),
                                           args.sent_combine_method,
                                           modeling_layer, dropout=args.dropout)
            d_inp_classifier = 4 * d_hid_model
        else:
            raise ValueError("Pair classifier type not found!")
        return pair_encoder, d_inp_classifier

    if args.shared_pair_enc:
        if not hasattr(model, "pair_encoder"):
            pair_encoder, d_inp_classifier = build_pair_encoder()
            model.pair_encoder = pair_encoder
        else:
            d_inp_classifier = 4 * d_inp if args.pair_enc == 'simple' else 4 * d_inp
        module = build_classifier(task, d_inp_classifier, args)
    else:
        pair_encoder, d_inp_classifier = build_pair_encoder()
        classifier = build_classifier(task, d_inp_classifier, args)
        module = nn.Sequential(pair_encoder, classifier)
    return module


def build_regressor(task, d_inp, args, model=None, vocab=None):
    ''' Build a task specific regressor '''
    cls_type, dropout, d_hid = \
        args.classifier, args.classifier_dropout, args.classifier_hid_dim
    if isinstance(task, STSBTask) or cls_type == 'log_reg':
        regressor = nn.Linear(d_inp, 1)
    elif isinstance(task, JOCITask):
        pair_encoder = SimplePairEncoder(vocab)
        model.pair_encoder = pair_encoder
        regressor = nn.Linear(d_inp, 1)
    elif cls_type == 'mlp':
        regressor = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid),
                                  nn.Tanh(), nn.LayerNorm(d_hid), nn.Dropout(p=dropout),
                                  nn.Linear(d_hid, 1))
    elif cls_type == 'fancy_mlp':
        regressor = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_inp, d_hid),
                                  nn.Tanh(), nn.LayerNorm(d_hid), nn.Dropout(p=dropout),
                                  nn.Linear(d_hid, d_hid), nn.Tanh(), nn.LayerNorm(d_hid),
                                  nn.Dropout(p=dropout), nn.Linear(d_hid, 1))
    return regressor


def build_lm(task, d_inp, args):
    ''' Build LM components (just map hidden states to vocab logits) '''
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
        elif isinstance(task, PairOrdinalRegressionTask):
            out = self._pair_regression_forward(batch, task)
        elif isinstance(task, LanguageModelingTask):
            out = self._lm_forward(batch, task)
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
        sent_emb = combine_hidden_states(sent_embs, sent_mask, self.combine_method)

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
        if hasattr(self, "pair_encoder"):
            pair_emb = self.pair_encoder(s1, s2, s1_mask, s2_mask)
            classifier = getattr(self, "%s_mdl" % task.name)
            logits = classifier(pair_emb)
        else:
            classifier = getattr(self, "%s_mdl" % task.name)
            logits = classifier(s1, s2, s1_mask, s2_mask)

        if 'labels' in batch:
            labels = batch['labels'].squeeze(-1)
            if isinstance(task, JOCITask):
                task.scorer1(mean_squared_error(logits, labels))
                task.scorer2(spearmanr(logits, labels)[0])
            else:
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
        if hasattr(self, "pair_encoder"):
            pair_emb = self.pair_encoder(s1, s2, s1_mask, s2_mask)
            classifier = getattr(self, "%s_mdl" % task.name)
            logits = classifier(pair_emb)
        else:
            classifier = getattr(self, "%s_mdl" % task.name)
            logits = classifier(s1, s2, s1_mask, s2_mask)

        # pass to a task specific classifier
        regressor = getattr(self, "%s_mdl" % task.name)
        scores = regressor(pair_emb)  # might want to pass sent_embs

        out['logits'] = scores  # maybe change the name here?
        if 'labels' in batch:
            labels = batch['labels']
            out['loss'] = F.mse_loss(scores, labels)
            if isinstance(task, STSBTask):
                scores = scores.squeeze(-1).data.cpu().numpy()
                labels = labels.squeeze(-1).data.cpu().numpy()
                task.scorer1(pearsonr(scores, labels)[0])
                task.scorer2(spearmanr(scores, labels)[0])
            elif isinstance(task, JOCITask):
                scores = scores.squeeze(-1).data.cpu().numpy()
                labels = labels.squeeze(-1).data.cpu().numpy()
                task.scorer1(mean_squared_error(scores, labels))
                task.scorer2(spearmanr(scores, labels)[0])
        return out

    def _seq_gen_forward(self, batch, task):
        ''' For translation, denoising, maybe language modeling? '''
        out = {}
        b_size, seq_len = batch['inputs']['words'].size()
        sent, sent_mask = self.sent_encoder(batch['inputs'])

        if 'targs' in batch:
            pass
        return out

    def _lm_forward(self, batch, task):
        ''' For language modeling? '''
        out = {}
        b_size, seq_len = batch['input']['words'].size()
        sent_encoder = self.sent_encoder

        if not isinstance(sent_encoder, BiLMEncoder):
            sent, mask = sent_encoder(batch['input'])
            sent = sent.masked_fill(1 - mask.byte(), 0)  # avoid NaNs
            hid2voc = getattr(self, "%s_hid2voc" % task.name)
            logits = hid2voc(sent).view(b_size * seq_len, -1)
            out['logits'] = logits
            targs = batch['targs']['words'].view(-1)
        else:
            sent, mask = sent_encoder(batch['input'], batch['input_bwd'])
            sent = sent.masked_fill(1 - mask.byte(), 0)  # avoid NaNs
            split = int(self.sent_encoder.output_dim / 2)
            fwd, bwd = sent[:, :, :split], sent[:, :, split:]
            hid2voc = getattr(self, "%s_hid2voc" % task.name)
            logits_fwd = hid2voc(fwd).view(b_size * seq_len, -1)
            logits_bwd = hid2voc(bwd).view(b_size * seq_len, -1)
            logits = torch.cat([logits_fwd, logits_bwd], dim=0)
            out['logits'] = logits
            trg_fwd = batch['targs']['words'].view(-1)
            trg_bwd = batch['targs_b']['words'].view(-1)
            targs = torch.cat([trg_fwd, trg_bwd])

        pad_idx = self.vocab.get_token_index(self.vocab._padding_token)
        out['loss'] = F.cross_entropy(logits, targs, ignore_index=pad_idx)
        task.scorer1(out['loss'].item())
        return out

    def _ranking_forward(self, batch, task):
        ''' For caption and image ranking '''
        raise NotImplementedError
