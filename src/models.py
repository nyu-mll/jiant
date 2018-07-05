'''Core model and functions for building it.'''
import sys
import math
import copy
import logging as log
import os

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error

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
from allennlp.training.metrics import Average

from utils import get_batch_utilization

from tasks import STSBTask, CoLATask, SSTTask, \
    PairClassificationTask, SingleClassificationTask, \
    PairRegressionTask, RankingTask, \
    SequenceGenerationTask, LanguageModelingTask, \
    PairOrdinalRegressionTask, JOCITask, WeakGroundedTask, \
    GroundedTask, MTTask, RedditTask
from modules import SentenceEncoder, BoWSentEncoder, \
    AttnPairEncoder, MaskedStackedSelfAttentionEncoder, \
    BiLMEncoder, ElmoCharacterEncoder, Classifier, Pooler, \
    SingleClassifier, PairClassifier, CNNEncoder
from utils import assert_for_log
from seq2seq_decoder import Seq2SeqDecoder


# Elmo stuff
# Look in $ELMO_SRC_DIR (e.g. /usr/share/jsalt/elmo) or download from web
ELMO_OPT_NAME = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHTS_NAME = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
ELMO_SRC_DIR = (os.getenv("ELMO_SRC_DIR") or
                "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/")
ELMO_OPT_PATH = os.path.join(ELMO_SRC_DIR, ELMO_OPT_NAME)
ELMO_WEIGHTS_PATH = os.path.join(ELMO_SRC_DIR, ELMO_WEIGHTS_NAME)
#  ELMO_OPT_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"  # pylint: disable=line-too-long
# ELMO_WEIGHTS_PATH =
# "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# # pylint: disable=line-too-long


def build_model(args, vocab, pretrained_embs, tasks):
    '''Build model according to args '''

    # Build embeddings.
    d_emb, embedder, cove_emb = build_embeddings(args, vocab, pretrained_embs)
    d_sent = args.d_hid

    # Build single sentence encoder: the main component of interest
    # Need special handling for language modeling
    tfm_params = Params({'input_dim': d_emb, 'hidden_dim': args.d_hid,
                         'projection_dim': args.d_tproj,
                         'feedforward_hidden_dim': args.d_ff,
                         'num_layers': args.n_layers_enc,
                         'num_attention_heads': args.n_heads})
    rnn_params = Params({'input_size': d_emb, 'bidirectional': args.bidirectional,
                         'hidden_size': args.d_hid, 'num_layers': args.n_layers_enc})

    if sum([isinstance(task, LanguageModelingTask) for task in tasks]):
        if args.bidirectional:
            rnn_params['bidirectional'] = False
            if args.sent_enc == 'rnn':
                fwd = s2s_e.by_name('lstm').from_params(copy.deepcopy(rnn_params))
                bwd = s2s_e.by_name('lstm').from_params(copy.deepcopy(rnn_params))
            elif args.sent_enc == 'transformer':
                fwd = MaskedStackedSelfAttentionEncoder.from_params(copy.deepcopy(tfm_params))
                bwd = MaskedStackedSelfAttentionEncoder.from_params(copy.deepcopy(tfm_params))
            sent_encoder = BiLMEncoder(vocab, embedder, args.n_layers_highway,
                                       fwd, bwd, dropout=args.dropout,
                                       skip_embs=args.skip_embs, cove_layer=cove_emb)
        else:  # not bidirectional
            if args.sent_enc == 'rnn':
                fwd = s2s_e.by_name('lstm').from_params(copy.deepcopy(rnn_params))
            elif args.sent_enc == 'transformer':
                fwd = MaskedStackedSelfAttentionEncoder.from_params(copy.deepcopy(tfm_params))
            sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                           fwd, skip_embs=args.skip_embs,
                                           dropout=args.dropout, cove_layer=cove_emb)
    elif args.sent_enc == 'bow':
        sent_encoder = BoWSentEncoder(vocab, embedder)
        d_sent = d_emb
    elif args.sent_enc == 'rnn':
        sent_rnn = s2s_e.by_name('lstm').from_params(copy.deepcopy(rnn_params))
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       sent_rnn, skip_embs=args.skip_embs,
                                       dropout=args.dropout, cove_layer=cove_emb)
        d_sent = (1 + args.bidirectional) * args.d_hid
    elif args.sent_enc == 'transformer':
        transformer = StackedSelfAttentionEncoder.from_params(copy.deepcopy(tfm_params))
        sent_encoder = SentenceEncoder(vocab, embedder, args.n_layers_highway,
                                       transformer, dropout=args.dropout,
                                       skip_embs=args.skip_embs, cove_layer=cove_emb)
    else:
        assert_for_log(False, "No valid sentence encoder specified.")

    d_sent += args.skip_embs * d_emb

    # Build model and classifiers
    model = MultiTaskModel(args, sent_encoder, vocab)
    build_modules(tasks, model, d_sent, vocab, embedder, args)
    if args.cuda >= 0:
        model = model.cuda()
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
        log.info("Loading ELMo from files:")
        log.info("ELMO_OPT_PATH = %s", ELMO_OPT_PATH)
        log.info("ELMO_WEIGHTS_PATH = %s", ELMO_WEIGHTS_PATH)
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
        task_params = get_task_specific_params(args, task.name)
        if isinstance(task, SingleClassificationTask):
            module = build_single_sentence_module(task, d_sent, task_params)
            setattr(model, '%s_mdl' % task.name, module)
        elif isinstance(task, (PairClassificationTask, PairRegressionTask,
                               PairOrdinalRegressionTask)):
            module = build_pair_sentence_module(task, d_sent, model, vocab,
                                                task_params)
            setattr(model, '%s_mdl' % task.name, module)
        elif isinstance(task, LanguageModelingTask):
            hid2voc = build_lm(task, d_sent, args)
            setattr(model, '%s_hid2voc' % task.name, hid2voc)
        elif isinstance(task, MTTask):
            decoder = Seq2SeqDecoder.from_params(vocab,
                                                 Params({'input_dim': d_sent,
                                                         'target_embedding_dim': 300,
                                                         'max_decoding_steps': 200,
                                                         'target_namespace': 'tokens',
                                                         'attention': 'bilinear',
                                                         'dropout': args.dropout,
                                                         'scheduled_sampling_ratio': 0.0}))
            setattr(model, '%s_decoder' % task.name, decoder)
        elif isinstance(task, SequenceGenerationTask):
            decoder, hid2voc = build_decoder(task, d_sent, vocab, embedder, args)
            setattr(model, '%s_decoder' % task.name, decoder)
            setattr(model, '%s_hid2voc' % task.name, hid2voc)
        elif isinstance(task, GroundedTask):
            task.img_encoder = CNNEncoder(model_name='resnet', path=task.path)
        elif isinstance(task, RankingTask):
            pooler, dnn_ResponseModel = build_reddit_module(task, d_sent, task_params)
            setattr(model, '%s_mdl' % task.name, pooler)
            setattr(model, '%s_Response_mdl' % task.name, dnn_ResponseModel) 

            #print("NEED TO ADD DNN to RESPONSE INPUT -- TO DO: IMPLEMENT QUICKLY")
        else:
            raise ValueError("Module not found for %s" % task.name)
    return


def get_task_specific_params(args, task):
    params = {}

    def get_task_attr(attr_name):
        return getattr(args, "%s_%s" % (task, attr_name)) if \
            hasattr(args, "%s_%s" % (task, attr_name)) else \
            getattr(args, attr_name)

    params['cls_type'] = get_task_attr("classifier")
    params['d_hid'] = get_task_attr("classifier_hid_dim")
    params['d_proj'] = get_task_attr("d_proj")
    params['shared_pair_attn'] = args.shared_pair_attn
    if args.shared_pair_attn:
        params['attn'] = args.pair_attn
        params['d_hid_attn'] = args.d_hid_attn
        params['dropout'] = args.classifier_dropout
    else:
        params['attn'] = get_task_attr("pair_attn")
        params['d_hid_attn'] = get_task_attr("d_hid_attn")
        params['dropout'] = get_task_attr("classifier_dropout")

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

    def forward(self, task, batch):
        '''
        Pass inputs to correct forward pass

        Args:
            - task
            - batch

        Returns:
            - out: dictionary containing task outputs and loss if label was in batch
        '''
        if 'input1' in batch and self.utilization is not None:
            self.utilization(get_batch_utilization(batch['input1']))
        if isinstance(task, SingleClassificationTask):
            out = self._single_sentence_forward(batch, task)
        elif isinstance(task, (PairClassificationTask, PairRegressionTask,
                               PairOrdinalRegressionTask)):
            out = self._pair_sentence_forward(batch, task)
        elif isinstance(task, LanguageModelingTask):
            out = self._lm_forward(batch, task)
        elif isinstance(task, SequenceGenerationTask):
            out = self._seq_gen_forward(batch, task)
        elif isinstance(task, GroundedTask):
            out = self._grounded_classification_forward(batch, task)
        elif isinstance(task, RankingTask):
            out = self._ranking_forward(batch, task)
        else:
            raise ValueError("Task-specific components not found!")
        return out

    def _single_sentence_forward(self, batch, task):
        out = {}

        # embed the sentence
        sent_embs, sent_mask = self.sent_encoder(batch['input1'])
        #pdb.set_trace()
        # pass to a task specific classifier
        classifier = getattr(self, "%s_mdl" % task.name)
        logits = classifier(sent_embs, sent_mask)

        if 'labels' in batch:
            labels = batch['labels'].squeeze(-1)
            out['loss'] = F.cross_entropy(logits, labels)
            if isinstance(task, CoLATask):
                task.scorer2(logits, labels)
                labels = labels.data.cpu().numpy()
                _, preds = logits.max(dim=1)
                task.scorer1(labels, preds.data.cpu().numpy())
            else:
                task.scorer1(logits, labels)
                if task.scorer2 is not None:
                    task.scorer2(logits, labels)
        out['logits'] = logits
        return out

    def _pair_sentence_forward(self, batch, task):
        out = {}

        # embed the sentence
        sent1, mask1 = self.sent_encoder(batch['input1'])
        sent2, mask2 = self.sent_encoder(batch['input2'])
        classifier = getattr(self, "%s_mdl" % task.name)
        logits = classifier(sent1, sent2, mask1, mask2)
        out['logits'] = logits

        if 'labels' in batch:
            labels = batch['labels'].squeeze(-1)
            if isinstance(task, JOCITask):
                logits = logits.squeeze(-1)
                out['loss'] = F.mse_loss(logits, labels)
                logits = logits.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                task.scorer1(mean_squared_error(logits, labels))
                task.scorer2(logits, labels)
            elif isinstance(task, STSBTask):
                logits = logits.squeeze(-1)
                out['loss'] = F.mse_loss(logits, labels)
                logits = logits.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                task.scorer1(logits, labels)
                task.scorer2(logits, labels)
            else:
                out['loss'] = F.cross_entropy(logits, labels)
                task.scorer1(logits, labels)
                if task.scorer2 is not None:
                    task.scorer2(logits, labels)
        return out


    def _ranking_forward(self, batch, task):
        ''' For caption and image ranking. This implementation is intended for Reddit'''
        out = {}
        # feed forwarding inputs through sentence encoders
        sent1, mask1 = self.sent_encoder(batch['input1'])  
        sent2, mask2 = self.sent_encoder(batch['input2']) 
        sent_pooler = getattr(self, "%s_mdl" % task.name) # pooler for both Input and Response
        sent_dnn = getattr(self, "%s_Response_mdl" % task.name) # dnn for Response  
        sent1_rep = sent_pooler(sent1, mask1)
        sent2_rep_pool = sent_pooler(sent2, mask2)
        sent2_rep = sent_dnn(sent2_rep_pool)

        if 1:
            #labels = batch['labels']
            #pdb.set_trace()
            sent1_rep = F.normalize(sent1_rep, 2, 1)
            sent2_rep = F.normalize(sent2_rep,2,1)

            # all the below implementation is binary cross entropy with weighted neg pairs
            # formula = sum(-log2(pos_pair_score) - scale * log2(1-neg_pair_score))

            # cosine similarity between every pair of samples
            cos_simi = torch.mm(sent1_rep, torch.transpose(sent2_rep, 0,1))
            cos_simi = F.sigmoid(cos_simi)  # bringing cos simi to [0,1]
            diag_elem = torch.diagonal(cos_simi)
            no_pos_pairs = len(diag_elem) 
            no_neg_pairs = no_pos_pairs * (no_pos_pairs - 1)

            #positive pairs loss: with the main diagonal elements
            pos_simi = torch.log2(diag_elem)  
            pos_loss = torch.neg(torch.sum(pos_simi)) 

            # negative pairs loss: with the off diagonal elements
            off_diag_elem = 1 - cos_simi + torch.diag(diag_elem)
            cos_simi_log = torch.log2(off_diag_elem)
            neg_loss = torch.neg(torch.sum(cos_simi_log))
            # scaling
            neg_loss_scaled = neg_loss * (no_pos_pairs/no_neg_pairs)

            out['loss'] = pos_loss + neg_loss_scaled
            #pdb.set_trace()
            # calculating accuracy
            pred = cos_simi.round()
          
            no_pos_pairs_correct = torch.trace(pred)
            # getting 1-pred and setting matrix with main diagonal elements to zero 
            offdiag_pred = torch.tril(1-pred, diagonal=-1) + torch.triu(1-pred, diagonal=1)
            no_neg_pairs_correct = torch.sum(offdiag_pred)
            
            total_correct = no_pos_pairs_correct + no_neg_pairs_correct
            batch_acc = total_correct.item()/(no_pos_pairs*no_pos_pairs)
            task.scorer1(batch_acc)
        return out
 

    def _seq_gen_forward(self, batch, task):
        ''' For translation, denoising, maybe language modeling? '''
        out = {}
        b_size, seq_len = batch['inputs']['words'].size()
        sent, sent_mask = self.sent_encoder(batch['inputs'])

        if isinstance(task, MTTask):
            decoder = getattr(self, "%s_decoder" % task.name)
            out = decoder.forward(sent, sent_mask, batch['targs'])
            task.scorer1(math.exp(out['loss'].item()))
            return out

        if 'targs' in batch:
            pass
        return out

    def _lm_forward(self, batch, task):
        ''' For language modeling? '''
        out = {}
        b_size, seq_len = batch['targs']['words'].size()
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

    def _grounded_classification_forward(self, batch, task):
        out = {}
        d_1, d_2 = self.sent_encoder.output_dim, 2048

        # embed the sentence, embed the image, map and classify
        sent_emb, sent_mask = self.sent_encoder(batch['input1'])
        image_map = nn.Linear(d_1, d_2).cuda()
        sent_transform = image_map(sent_emb)
        ids = batch['ids'].cpu().squeeze(-1)
        ids = list(ids.data.numpy())
        labels = batch['labels'].cpu().squeeze(-1)
        labels = [int(item) for item in labels.data.numpy()]

        seq, true = [], []
        for i in range(len(ids)):
            img_id, label = ids[i], labels[i]
            init_emb = task.img_encoder.forward(int(img_id)).data.numpy()[0]
            seq.append(torch.tensor(init_emb, dtype=torch.float))
            true.append(label)
        img_emb = torch.stack(seq, dim=0)

        batch_size = len(labels)
        sent_transform = sent_transform.view(batch_size, -1)
        image_map = nn.Linear(list(sent_transform.size())[-1], d_2).cuda()
        sent_transform = image_map(sent_transform)

        '''
        cos = nn.SmoothL1Loss()
        cos = nn.MSELoss()
        cos = nn.L1Loss()
        out['loss'] = cos(sent_emb, torch.tensor(img_emb, requires_grad=False))
        '''
        cos = nn.CosineEmbeddingLoss()
        flags = Variable(torch.ones(len(labels)))
        out['loss'] = cos(
            torch.tensor(
                sent_transform, dtype=torch.float), torch.tensor(
                img_emb, dtype=torch.float), flags)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = cos(
            torch.tensor(
                sent_transform,
                dtype=torch.float),
            torch.tensor(
                img_emb,
                dtype=torch.float))
        classifier = nn.Linear(len(labels), len(labels))
        logits = classifier(sim)
        out['logits'] = logits

        preds = [1 if item > 0 else 0 for item in logits.data.numpy()]
        acc = [1 if preds[i] == labels[i] else 0 for i in range(len(labels))]
        task.scorer1.__call__(np.sum(acc) / len(acc))

        return out
