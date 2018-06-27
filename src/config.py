'''Train a multi-task model using AllenNLP '''
import os
import sys
import time
import random
import types
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import argparse
import pyhocon
import json

import hocon_writer

class Params(object):
    """Params handler object.

    This functions as a nested dict, but allows for seamless dot-style access, similar to tf.HParams but without the type validation. For example:

    p = Params(name="model", num_layers=4)
    p.name  # "model"
    p['data'] = Params(path="file.txt")
    p.data.path  # "file.txt"
    """

    @staticmethod
    def clone(source, strict=True):
        if isinstance(source, pyhocon.ConfigTree):
            return Params(**source.as_plain_ordered_dict())
        elif isinstance(source, Params):
            return Params(**source.as_dict())
        elif isinstance(source, dict):
            return Params(**source)
        elif strict:
            raise ValueError("Cannot clone from type: " + str(type(source)))
        else:
            return None

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        assert isinstance(k, str)
        if isinstance(self.get(k, None), types.FunctionType):
            raise ValueError("Invalid parameter name (overrides reserved name '%s')." % k)

        converted_val = Params.clone(v, strict=False)
        if converted_val is not None:
            setattr(self, k, converted_val)
        else:  # plain old data
            setattr(self, k, v)
        self._known_keys.add(k)

    def __init__(self, **kw):
        """Create from a list of key-value pairs."""
        self._known_keys = set()
        for k,v in kw.items():
            self[k] = v

    def get(self, k, default):
        return getattr(self, k, default)

    def keys(self):
        return list(self._known_keys)

    def as_dict(self):
        """Recursively convert to a plain dict."""
        convert = lambda v: v.as_dict() if isinstance(v, Params) else v
        return {k:convert(self[k]) for k in self.keys()}

    def __repr__(self):
        return self.as_dict().__repr__()

    def __str__(self):
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

# Argument handling is as follows:
# 1) read config file into pyhocon.ConfigTree
# 2) merge overrides into the ConfigTree
# 3) validate specific parameters with custom logic
def params_from_file(config_file, overrides=None):
    with open(config_file) as fd:
        config_string = fd.read()
    if overrides:
        # Append overrides to file to allow for references and injection.
        config_string += "\n"
        config_string += overrides
    basedir = os.path.dirname(config_file)  # directory context for includes
    config = pyhocon.ConfigFactory.parse_string(config_string, basedir=basedir)
    return Params.clone(config)

def write_params(params, config_file):
    config = pyhocon.ConfigFactory.from_dict(params.as_dict())
    with open(config_file, 'w') as fd:
        fd.write(hocon_writer.HOCONConverter.to_hocon(config, indent=2))

def parse_arguments(arguments):
    parser = argparse.ArgumentParser(description='')

    # Logistics
    parser.add_argument('--cuda', help='-1 if no CUDA, else gpu id', type=int, default=0)
    parser.add_argument('--random_seed', help='random seed to use', type=int, default=19)

    # Paths and logging
    parser.add_argument('--log_file', help='file to log to', type=str, default='log.log')
    parser.add_argument('--data_dir', help='directory containing all data', type=str)
    parser.add_argument('--exp_dir', help='directory containing shared preprocessing', type=str)
    parser.add_argument('--run_dir', help='directory for saving results, models, etc.', type=str)

    # Time saving flags
    parser.add_argument('--should_train', help='1 if should train model', type=int, default=1)
    parser.add_argument('--load_model', help='1 if load from checkpoint', type=int, default=1)
    parser.add_argument('--force_load_epoch', help='Force loading from a certain epoch',
                        type=int, default=-1)
    parser.add_argument('--reload_tasks', help='1 if force re-reading of tasks', type=int,
                        default=0)
    parser.add_argument('--reload_indexing', help='1 if force re-indexing for all tasks',
                        type=int, default=0)
    parser.add_argument('--reload_vocab', help='1 if force vocabulary rebuild', type=int, default=0)

    # Tasks and task-specific modules
    parser.add_argument('--train_tasks', help='comma separated list of tasks, or "all" or "none"',
                        type=str)
    parser.add_argument('--eval_tasks', help='list of additional tasks to train a classifier,' +
                        'then evaluate on', type=str, default='')
    parser.add_argument('--train_for_eval', help='1 if models should be trained for the eval tasks (defaults to True)',
                        type=int, default=1)
    parser.add_argument('--classifier', help='type of classifier to use', type=str,
                        default='log_reg', choices=['log_reg', 'mlp', 'fancy_mlp'])
    parser.add_argument('--classifier_hid_dim', help='hid dim of classifier', type=int, default=512)
    parser.add_argument('--classifier_dropout', help='classifier dropout', type=float, default=0.0)
    parser.add_argument('--d_hid_dec', help='decoder hidden size', type=int, default=300)
    parser.add_argument('--n_layers_dec', help='n decoder layers', type=int, default=1)

    # Preprocessing options
    parser.add_argument('--max_seq_len', help='max sequence length', type=int, default=40)
    parser.add_argument('--max_word_v_size', help='max word vocab size', type=int, default=30000)
    parser.add_argument('--max_char_v_size', help='max char vocab size', type=int, default=250)

    # Embedding options
    parser.add_argument(
        '--word_embs',
        help='type of word embs to use',
        default='fastText',
        choices=[
            'none',
            'scratch',
            'glove',
            'fastText'])
    parser.add_argument('--word_embs_file', help='file containing word embs', type=str, default='')
    parser.add_argument('--fastText', help='1 if use fastText model', type=int, default=0)
    parser.add_argument('--fastText_model_file', help='file containing fastText model',
                        type=str, default=None)
    parser.add_argument('--d_word', help='dimension of word embeddings', type=int, default=300)
    parser.add_argument('--d_char', help='dimension of char embeddings', type=int, default=100)
    parser.add_argument('--n_char_filters', help='n char filters', type=int, default=100)
    parser.add_argument('--char_filter_sizes', help='filter sizes for char emb cnn', type=str,
                        default='2,3,4,5')
    parser.add_argument('--elmo', help='1 if use elmo', type=int, default=0)
    parser.add_argument('--deep_elmo', help='1 if use elmo post LSTM', type=int, default=0)
    parser.add_argument('--cove', help='1 if use cove', type=int, default=0)
    parser.add_argument('--char_embs', help='1 if use character embs', type=int, default=0)
    parser.add_argument('--dropout_embs', help='drop rate for embeddings', type=float, default=.2)
    parser.add_argument('--preproc_file', help='file containing saved preprocessing stuff',
                        type=str, default='preproc.pkl')

    # Model options
    parser.add_argument('--sent_enc', help='type of sent encoder to use', type=str, default='rnn',
                        choices=['bow', 'rnn', 'transformer', 'transformer-d'])
    parser.add_argument('--sent_combine_method', help='how to aggregate hidden states of sent rnn',
                        type=str, default='max', choices=['max', 'mean', 'final'])

    parser.add_argument('--shared_pair_enc', help='1 to share pair encoder for pair sentence tasks',
                        type=int, default=1)
    parser.add_argument('--bidirectional', help='1 if bidirectional RNN', type=int, default=1)
    parser.add_argument('--pair_enc', help='type of pair encoder to use', type=str,
                        default='simple', choices=['simple', 'attn'])
    parser.add_argument('--d_hid', help='hidden dimension size', type=int, default=512)
    parser.add_argument('--n_layers_enc', help='number of RNN layers', type=int, default=1)
    parser.add_argument('--n_layers_highway', help='num of highway layers', type=int, default=1)
    parser.add_argument('--n_heads', help='num of transformer heads', type=int, default=8)
    parser.add_argument('--d_proj', help='transformer projection dim', type=int, default=64)
    parser.add_argument('--d_ff', help='transformer feedforward dim', type=int, default=2048)
    parser.add_argument('--dropout', help='dropout rate to use in training', type=float, default=.2)

    # Training options
    parser.add_argument('--no_tqdm', help='1 to turn off tqdm', type=int, default=0)
    parser.add_argument('--trainer_type', help='type of trainer', type=str,
                        choices=['sampling', 'mtl'], default='sampling')
    parser.add_argument('--shared_optimizer', help='1 to use same optimizer for all tasks',
                        type=int, default=1)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument(
        '--optimizer',
        help='optimizer to use. all valid AllenNLP options are available, including `sgd`. `adam` uses to the newer AMSGrad variant.',
        type=str,
        default='sgd')
    parser.add_argument('--n_epochs', help='n epochs to train for', type=int, default=10)
    parser.add_argument('--lr', help='starting learning rate', type=float, default=1.0)
    parser.add_argument('--min_lr', help='minimum learning rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', help='max grad norm', type=float, default=5.)
    parser.add_argument('--weight_decay', help='weight decay value', type=float, default=0.0)
    parser.add_argument('--task_patience', help='patience in decaying per task lr',
                        type=int, default=0)
    parser.add_argument('--scheduler_threshold', help='scheduler threshold',
                        type=float, default=0.0)
    parser.add_argument('--lr_decay_factor', help='lr decay factor when val score doesn\'t improve',
                        type=float, default=.5)
    parser.add_argument('--warmup', help='n warmup steps for Transformer LR scheduler',
                        type=int, default=4000)

    # Multi-task training options
    parser.add_argument('--val_interval', help='Number of passes between validation checks',
                        type=int, default=10)
    parser.add_argument('--max_vals', help='Maximum number of validation checks', type=int,
                        default=100)
    parser.add_argument('--bpp_base', help='Number of batches to train on per sampled task',
                        type=int, default=1)
    parser.add_argument('--weighting_method', help='Weighting method for sampling', type=str,
                        choices=['uniform', 'proportional'], default='uniform')
    parser.add_argument('--scaling_method', help='method for scaling loss', type=str,
                        choices=['min', 'max', 'unit', 'none'], default='none')
    parser.add_argument('--patience', help='patience in early stopping', type=int, default=5)

    # Evaluation options
    parser.add_argument(
        '--eval_val_interval',
        help='val interval for eval task',
        type=int,
        default=1000)
    parser.add_argument('--eval_max_vals', help='Maximum number of validation checks for eval task',
                        type=int, default=100)
    parser.add_argument('--write_preds', help='1 if write test predictions', type=int, default=1)

    args = parser.parse_args(arguments)

    return args


