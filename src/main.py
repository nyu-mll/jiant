'''Train a multi-task model using AllenNLP '''
import os
import sys
import time
import random
import argparse
import logging as log
import ipdb as pdb
import torch

from preprocess import build_tasks
from models import build_model
from trainer import build_trainer
from evaluate import evaluate, load_model_state, write_results, write_preds

import _pickle as pkl

def main(arguments):
    ''' Train or load a model. Evaluate on some tasks. '''
    parser = argparse.ArgumentParser(description='')

    # Logistics
    parser.add_argument('--cuda', help='-1 if no CUDA, else gpu id', type=int, default=0)
    parser.add_argument('--random_seed', help='random seed to use', type=int, default=19)

    # Paths and logging
    parser.add_argument('--log_file', help='file to log to', type=str, default='log.log')
    parser.add_argument('--data_dir', help='directory containing all data', type=str,
                        default='/misc/vlgscratch4/BowmanGroup/awang/processed_data/mtl-sentence-representations/')
    parser.add_argument('--exp_dir', help='directory containing shared preprocessing', type=str)
    parser.add_argument('--run_dir', help='directory for saving results, models, etc.', type=str)
    parser.add_argument('--word_embs_file', help='file containing word embs', type=str, default='')
    parser.add_argument('--path_to_cove', help='path to cove repo', type=str,
                        default='/misc/vlgscratch4/BowmanGroup/awang/models/cove')
    parser.add_argument('--preproc_file', help='file containing saved preprocessing stuff',
                        type=str, default='preproc.pkl')

    # Time saving flags
    parser.add_argument('--should_train', help='1 if should train model', type=int, default=1)
    parser.add_argument('--load_model', help='1 if load from checkpoint', type=int, default=1)
    parser.add_argument('--force_load_epoch', help='Force loading from a certain epoch',
                        type=int, default=-1)
    parser.add_argument('--reload_tasks', help='1 if force re-reading of tasks', type=int, default=0)
    parser.add_argument('--reload_indexing', help='1 if force re-indexing for all tasks',
                        type=int, default=0)
    parser.add_argument('--reload_vocab', help='1 if force vocabulary rebuild', type=int, default=0)

    # Tasks and task-specific modules
    parser.add_argument('--train_tasks', help='comma separated list of tasks, or "all" or "none"',
                        type=str)
    parser.add_argument('--eval_tasks', help='list of additional tasks to train a classifier,' +
                        'then evaluate on', type=str, default='')
    parser.add_argument('--classifier', help='type of classifier to use', type=str,
                        default='log_reg', choices=['log_reg', 'mlp', 'fancy_mlp'])
    parser.add_argument('--classifier_hid_dim', help='hid dim of classifier', type=int, default=512)
    parser.add_argument('--classifier_dropout', help='classifier dropout', type=float, default=0.0)
    parser.add_argument('--d_hid_dec', help='decoder hidden size', type=int, default=300)
    parser.add_argument('--n_layers_dec', help='n decoder layers', type=int, default=1)

    # Preprocessing options
    parser.add_argument('--max_seq_len', help='max sequence length', type=int, default=40)
    parser.add_argument('--max_word_v_size', help='max word vocab size', type=int, default=30000)

    # Embedding options
    parser.add_argument('--no_word_embs', help='1 if no word embs', type=int, default=0)
    parser.add_argument('--glove', help='1 if use glove', type=int, default=1)
    parser.add_argument('--fasttext', help='1 if use fasttext', type=int, default=0)
    parser.add_argument('--word_embs', help='type of word embeddings', type=str,
                        choices=['glove', 'fasttext', 'scratch', 'none'], default='glove')
    parser.add_argument('--dropout_embs', help='drop rate for embeddings', type=float, default=.2)
    parser.add_argument('--d_word', help='dimension of word embeddings', type=int, default=300)
    parser.add_argument('--train_words', help='1 if make word embs trainable', type=int, default=0)
    parser.add_argument('--elmo', help='1 if use elmo', type=int, default=0)
    parser.add_argument('--deep_elmo', help='1 if use elmo post LSTM', type=int, default=0)
    parser.add_argument('--cove', help='1 if use cove', type=int, default=0)

    # Model options
    parser.add_argument('--sent_enc', help='type of sent encoder to use', type=str, default='rnn',
                        choices=['bow', 'rnn'])
    parser.add_argument('--shared_pair_enc', help='1 to share pair encoder for pair sentence tasks',
                        type=int, default=1)
    parser.add_argument('--pair_enc', help='type of pair encoder to use', type=str,
                        default='simple', choices=['simple', 'attn'])
    parser.add_argument('--d_hid', help='hidden dimension size', type=int, default=4096)
    parser.add_argument('--n_layers_enc', help='number of RNN layers', type=int, default=1)
    parser.add_argument('--n_layers_highway', help='num of highway layers', type=int, default=1)
    parser.add_argument('--dropout', help='dropout rate to use in training', type=float, default=.2)

    # Training options
    parser.add_argument('--no_tqdm', help='1 to turn off tqdm', type=int, default=0)
    parser.add_argument('--trainer_type', help='type of trainer', type=str,
                        choices=['sampling', 'mtl'], default='sampling')
    parser.add_argument('--shared_optimizer', help='1 to use same optimizer for all tasks',
                        type=int, default=1)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--optimizer', help='optimizer to use. all valid AllenNLP options are available, including `sgd`. `adam` uses to the newer AMSGrad variant.', type=str, default='sgd')
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

    # Multi-task training options
    parser.add_argument('--val_interval', help='Number of passes between validation checks',
                        type=int, default=10)
    parser.add_argument('--max_vals', help='Maximum number of validation checks', type=int,
                        default=100)
    parser.add_argument('--bpp_method', help='if using nonsampling trainer, ' +
                        'method for calculating number of batches per pass', type=str,
                        choices=['fixed', 'percent_tr', 'proportional_rank'], default='fixed')
    parser.add_argument('--bpp_base', help='If sampling or fixed bpp' +
                        'per pass, this is the bpp. If proportional, this ' +
                        'is the smallest number', type=int, default=10)
    parser.add_argument('--weighting_method', help='Weighting method for sampling', type=str,
                        choices=['uniform', 'proportional'], default='uniform')
    parser.add_argument('--scaling_method', help='method for scaling loss', type=str,
                        choices=['min', 'max', 'unit', 'none'], default='none')
    parser.add_argument('--patience', help='patience in early stopping', type=int, default=5)
    parser.add_argument('--task_ordering', help='Method for ordering tasks', type=str, default='given',
                        choices=['given', 'random', 'random_per_pass', 'small_to_large', 'large_to_small'])

    parser.add_argument('--write_preds', help='1 if write test preditions', type=int, default=1)

    args = parser.parse_args(arguments)

    # Logistics #
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO, datefmt='%m/%d %I:%M:%S %p')
    log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir, args.log_file)))
    log.info(args)
    seed = random.randint(1, 10000) if args.random_seed < 0 else args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda >= 0:
        log.info("Using GPU %d", args.cuda)
        torch.cuda.set_device(args.cuda)
        torch.cuda.manual_seed_all(seed)
    log.info("Using random seed %d", seed)

    # Prepare data #
    log.info("Loading tasks...")
    start_time = time.time()
    train_tasks, eval_tasks, vocab, word_embs = build_tasks(args)
    tasks = train_tasks + eval_tasks
    log.info('\tFinished loading tasks in %.3fs', time.time() - start_time)

    # Build model #
    log.info('Building model...')
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    log.info('\tFinished building model in %.3fs', time.time() - start_time)

    # Train on train tasks #
    if train_tasks and args.should_train:
        log.info("Training...")
        trainer, _, opt_params, schd_params = build_trainer(args, model)
        to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        best_epochs = trainer.train(train_tasks, args.val_interval, args.bpp_base,
                                    args.weighting_method, args.scaling_method,
                                    to_train, opt_params, schd_params,
                                    args.shared_optimizer, args.load_model)
    else:
        log.info("Skipping training.")
        best_epochs = {}

    # Select model checkpoint from training to load
    if args.force_load_epoch >= 0: # force loading a particular epoch
        epoch_to_load = args.force_load_epoch
    elif "macro" in best_epochs:
        epoch_to_load = best_epochs['macro']
    else:
        serialization_files = os.listdir(args.run_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        if model_checkpoints:
            epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th")) \
                                 for x in model_checkpoints])
        else:
            epoch_to_load = -1
    if epoch_to_load >= 0:
        state_path = os.path.join(args.run_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        load_model_state(model, state_path, args.cuda)

    # Train just the task-specific components for eval tasks
    # TODO(Alex): currently will overwrite model checkpoints from training
    for task in eval_tasks:
        pred_module = getattr(model, "%s_mdl" % task.name)
        to_train = [(n, p) for n, p in pred_module.named_parameters() if p.requires_grad]
        trainer, _, opt_params, schd_params = build_trainer(args, model)
        best_epoch = trainer.train([task], args.val_interval, args.bpp_base,
                                   args.weighting_method, args.scaling_method,
                                   to_train, opt_params, schd_params,
                                   args.shared_optimizer, args.load_model)
        best_epoch = best_epoch[task.name]
        layer_path = os.path.join(args.run_dir, "model_state_epoch_{}.th".format(best_epoch))
        load_model_state(model, layer_path, args.cuda)

    # Evaluate #
    log.info("Evaluating...")
    _, te_preds = evaluate(model, tasks, args.batch_size, args.cuda, "test")
    val_results, _ = evaluate(model, tasks, args.batch_size, args.cuda, "val")
    if args.write_preds:
        write_preds(te_preds, args.run_dir)
    write_results(val_results, os.path.join(args.exp_dir, "results.tsv"),
                  args.run_dir.split('/')[-1])

    log.info("Done!")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
