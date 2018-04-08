'''Train a multi-task model using AllenNLP'''
import os
import sys
import time
import copy
import random
import argparse
import logging as log
import _pickle as pkl
import ipdb as pdb
import torch

from allennlp.data.iterators import BasicIterator
from util import device_mapping

from preprocess import build_tasks
from models import build_model
from trainer import MultiTaskTrainer, build_trainer
from evaluate import evaluate

def main(arguments):
    ''' Train or load a model. Evaluate on some tasks. '''
    parser = argparse.ArgumentParser(description='')

    # Logistics
    parser.add_argument('--cuda', help='0 if no CUDA, else gpu id', type=int, default=0)
    parser.add_argument('--random_seed', help='random seed to use', type=int, default=19)

    # Paths and logging
    parser.add_argument('--log_file', help='path to log to', type=str, default=0)
    parser.add_argument('--exp_dir', help='experiment directory containing shared preprocessing',
                        type=str, default='')
    parser.add_argument('--run_dir', help='run directory for saving results, models, etc.',
                        type=str, default='')
    parser.add_argument('--vocab_path', help='folder containing vocab stuff', type=str, default='')
    parser.add_argument('--word_embs_file', help='file containing word embs', type=str, default='')
    parser.add_argument('--preproc_file', help='file containing saved preprocessing stuff',
                        type=str, default='')

    # Time saving flags
    parser.add_argument('--should_train', help='1 if should train model', type=int, default=1)
    parser.add_argument('--load_model', help='1 if load from checkpoint', type=int, default=1)
    parser.add_argument('--load_tasks', help='1 if load tasks', type=int, default=1)
    parser.add_argument('--load_preproc', help='1 if load vocabulary', type=int, default=1)

    # Tasks and task-specific classifiers
    parser.add_argument('--train_tasks', help='comma separated list of tasks, or "all" or "none"',
                        type=str)
    parser.add_argument('--eval_tasks', help='list of additional tasks to train a classifier,' +
                        'then evaluate on', type=str, default='')
    parser.add_argument('--classifier', help='type of classifier to use', type=str,
                        default='log_reg', choices=['log_reg', 'mlp', 'fancy_mlp'])
    parser.add_argument('--classifier_hid_dim', help='hid dim of classifier', type=int, default=512)
    parser.add_argument('--classifier_dropout', help='classifier dropout', type=float, default=0.0)

    # Preprocessing options
    parser.add_argument('--max_seq_len', help='max sequence length', type=int, default=100)
    parser.add_argument('--max_word_v_size', help='max word vocabulary size', type=int, default=50000)
    parser.add_argument('--max_char_v_size', help='char vocabulary size', type=int, default=999)

    # Embedding options
    parser.add_argument('--d_char', help='dimension of char embeddings', type=int, default=100)
    parser.add_argument('--char_encoder', help='char embedding encoder', type=str, default='cnn',
                        choices=['bow', 'cnn'])
    parser.add_argument('--n_char_filters', help='num of conv filters for ' +
                        'char embedding combiner', type=int, default=64)
    parser.add_argument('--char_filter_sizes', help='filter sizes for char' +
                        ' embedding combiner', type=str, default='2,3,4,5')
    parser.add_argument('--dropout_embs', help='dropout rate for embeddisn', type=float, default=.2)
    parser.add_argument('--d_word', help='dimension of word embeddings', type=int, default=300)
    parser.add_argument('--train_words', help='1 if make word embs trainable', type=int, default=1)
    parser.add_argument('--elmo', help='1 if use elmo', type=int, default=0)
    parser.add_argument('--deep_elmo', help='1 if use elmo post LSTM', type=int, default=0)
    parser.add_argument('--elmo_no_glove', help='1 if don\'t use glove with elmo', type=int, default=0)
    parser.add_argument('--cove', help='1 if use cove', type=int, default=0)

    # Model options
    parser.add_argument('--pair_enc', help='type of pair encoder to use', type=str, default='bidaf',
                        choices=['simple', 'bidaf', 'bow', 'attn'])
    parser.add_argument('--d_hid', help='hidden dimension size', type=int, default=4096)
    parser.add_argument('--n_layers_enc', help='number of RNN layers', type=int, default=1)
    parser.add_argument('--n_layers_highway', help='num of highway layers', type=int, default=1)
    parser.add_argument('--dropout', help='dropout rate to use in training', type=float, default=.2)

    # Training options
    parser.add_argument('--no_tqdm', help='1 to turn off tqdm', type=int, default=0)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('--optimizer', help='optimizer to use', type=str, default='sgd')
    parser.add_argument('--n_epochs', help='n epochs to train for', type=int, default=10)
    parser.add_argument('--lr', help='starting learning rate', type=float, default=1.0)
    parser.add_argument('--min_lr', help='minimum learning rate', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', help='max grad norm', type=float, default=5.)
    parser.add_argument('--weight_decay', help='weight decay value', type=float, default=0.0)
    parser.add_argument('--task_patience', help='patience in decaying per task lr',
                        type=int, default=0)
    parser.add_argument('--scheduler_threshold', help='scheduler threshold',
                        type=float, default=0.0)
    parser.add_argument('--lr_decay_factor', help='lr decay factor when val score' +
                        ' doesn\'t improve', type=float, default=.5)

    # Multi-task training options
    parser.add_argument('--val_interval', help='Number of passes between ' +
                        ' validating', type=int, default=10)
    parser.add_argument('--max_vals', help='Maximum number of validation checks', type=int,
                        default=100)
    parser.add_argument('--bpp_method', help='How to calculate ' +
                        'the number of batches per pass for each task', type=str,
                        choices=['fixed', 'percent_tr', 'proportional_rank'],
                        default='fixed')
    parser.add_argument('--bpp_base', help='If fixed n batches ' +
                        'per pass, this is the number. If proportional, this ' +
                        'is the smallest number', type=int, default=10)
    parser.add_argument('--patience', help='patience in early stopping', type=int, default=5)
    parser.add_argument('--task_ordering', help='Method for ordering tasks', type=str, default='given',
                        choices=['given', 'random', 'random_per_pass', 'small_to_large', 'large_to_small'])

    args = parser.parse_args(arguments)

    # Logistics #
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO, datefmt='%m/%d %I:%M:%S %p')
    print(args.log_file)
    file_handler = log.FileHandler(args.log_file)
    log.getLogger().addHandler(file_handler)
    log.info(args)
    seed = random.randint(1, 10000) if args.random_seed < 0 else args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda >= 0:
        log.info("Using GPU %d", args.cuda)
        torch.cuda.set_device(args.cuda)
        torch.cuda.manual_seed_all(seed)
    log.info("Using random seed %d", seed)

    # Load tasks #
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

    # Set up trainer #
    iterator = BasicIterator(args.batch_size)
    trainer, train_params, opt_params, schd_params = build_trainer(args, model, iterator)

    # Train #
    if train_tasks and args.should_train:
        to_train = [p for p in model.parameters() if p.requires_grad]
        best_epochs = trainer.train(train_tasks, args.task_ordering, args.val_interval,
                                    args.max_vals, args.bpp_method, args.bpp_base, to_train,
                                    opt_params, schd_params, args.load_model)
    else:
        log.info("Skipping training.")
        best_epochs = {}

    # train just the classifiers for eval tasks
    for task in eval_tasks:
        pred_layer = getattr(model, "%s_pred_layer" % task.name)
        to_train = pred_layer.parameters()
        trainer = MultiTaskTrainer.from_params(model, args.run_dir + '/%s/' % task.name,
                                               iterator, copy.deepcopy(train_params))
        trainer.train([task], args.task_ordering, 1, args.max_vals, 'percent_tr', 1, to_train,
                      opt_params, schd_params, 1)
        layer_path = os.path.join(args.run_dir, task.name, "%s_best.th" % task.name)
        layer_state = torch.load(layer_path, map_location=device_mapping(args.cuda))
        model.load_state_dict(layer_state)

    # Evaluate: load the different task best models and evaluate them
    # TODO(Alex): put this in evaluate file
    log.info('***** TEST RESULTS *****')
    all_results = {}

    if best_epochs is None:
        serialization_files = os.listdir(args.run_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th")) \
                             for x in model_checkpoints])
    else:
        epoch_to_load = -1

    for task in [task.name for task in train_tasks] + ['micro', 'macro']:
        #model_path = os.path.join(args.run_dir, "%s_best.th" % task)
        load_idx = best_epochs[task] if best_epochs is not None else epoch_to_load
        model_path = os.path.join(args.run_dir,
                                  "model_state_epoch_{}.th".format(load_idx))

        model_state = torch.load(model_path, map_location=device_mapping(args.cuda))
        model.load_state_dict(model_state)
        te_results, te_preds = evaluate(model, tasks, iterator, cuda_device=args.cuda, split="test")
        val_results, val_preds = evaluate(model, tasks, iterator, cuda_device=args.cuda, split="val")
        all_metrics_str = ', '.join(['%s: %.5f' % (metric, score) for metric, score in \
                                     te_results.items()])
        log.info('%s, %s', task, all_metrics_str)
        all_results[task] = (val_results, te_results, model_path)
    results_file = os.path.join(args.run_dir, "results.pkl")
    pkl.dump(all_results, open(results_file, 'wb'))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
