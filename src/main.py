'''Train a multi-task model using AllenNLP '''
import os
import sys
import time
import random
import argparse
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

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
    parser.add_argument('--data_dir', help='directory containing all data', type=str)
    parser.add_argument('--exp_dir', help='directory containing shared preprocessing', type=str)
    parser.add_argument('--run_dir', help='directory for saving results, models, etc.', type=str)

    # Time saving flags
    parser.add_argument('--load_model_tmp', help='Path from which to load from checkpoint; None to start training from scratch', type=str, default="")
    parser.add_argument('--reload_tasks', help='1 if force re-reading of tasks', type=int,
                        default=0)
    parser.add_argument('--reload_indexing', help='1 if force re-indexing for all tasks',
                        type=int, default=0)
    parser.add_argument('--reload_vocab', help='1 if force vocabulary rebuild', type=int, default=0)

    # Control flow for main
    parser.add_argument('--do_train', help='1 to run train else 0', type=int, default=0)
    parser.add_argument('--do_eval', help='1 to run eval tasks (where model can be retrained for eval task) else 0', type=int, default=0)
    parser.add_argument('--do_probe', help='1 to run test-only probing tasks else 0', type=int, default=0)
    parser.add_argument('--train_for_eval', help='1 if models should be trained for the eval tasks else 0', type=int, default=0)

    # Tasks and task-specific modules
    parser.add_argument('--train_tasks', help='comma separated list of tasks, or "all" or "none"',
                        type=str)
    parser.add_argument('--eval_tasks', help='list of additional tasks to train a classifier,' +
                        'then evaluate on', type=str, default='')
    parser.add_argument('--probing_tasks', help='list of additional tasks to test on (no retraining)', type=str, default='')
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
    parser.add_argument('--d_hid', help='hidden dimension size', type=int, default=4096)
    parser.add_argument('--n_layers_enc', help='number of RNN layers', type=int, default=1)
    parser.add_argument('--n_layers_highway', help='num of highway layers', type=int, default=1)
    parser.add_argument('--n_heads', help='num of transformer heads', type=int, default=1)
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
    parser.add_argument('--eval_val_interval', help='val interval for eval task', type=int, default=1000)
    parser.add_argument('--eval_max_vals', help='Maximum number of validation checks for eval task',
                        type=int, default=100)
    parser.add_argument('--write_preds', help='1 if write test predictions', type=int, default=1)

    args = parser.parse_args(arguments)

    # Logistics #
    log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir, args.log_file)))
    log.info(args)
    seed = random.randint(1, 10000) if args.random_seed < 0 else args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda >= 0:
        log.info("Using GPU %d", args.cuda)
        try:
            torch.cuda.set_device(args.cuda)
            torch.cuda.manual_seed_all(seed)
        except AttributeError:
            log.warning(
                "GPU access failed. You might be using a CPU-only installation of PyTorch. Falling back to CPU.")
            args.cuda = -1
    log.info("Using random seed %d", seed)
      
    # Prepare data #
    log.info("Loading tasks...")
    start_time = time.time()
    train_tasks, eval_tasks, vocab, word_embs = build_tasks(args)
    tasks = train_tasks + eval_tasks
    log.info('\tFinished loading tasks in %.3fs', time.time() - start_time)

    # Build or load model #
    log.info('Building model...')
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    log.info('\tFinished building model in %.3fs', time.time() - start_time)

    steps_log = []
    # Check that necessary parameters are set for each step. Exit with error if not.
    if args.load_model_tmp:
      try:
        assert os.path.exists(args.load_model_tmp)
      except AssertionError:
        log.error("Error: Attempting to load model from non-existent path: [%s]"%args.load_model_tmp)
        return 0
      steps_log.append("Loading model from path: %s"%args.load_model_tmp)
    else:
      steps_log.append("Initializing model from scratch.")

    if args.do_train:
      try:
        assert args.train_tasks
      except AssertionError:
        log.error("Error: Must specify at least on training task: [%s]"%args.train_tasks)
        return 0 
      steps_log.append("Training model on tasks: %s"%args.train_tasks)

    if args.train_for_eval:
     steps_log.append("Re-training model for individual eval tasks")

    if args.do_eval:
      try:
        assert args.eval_tasks
      except AssertionError:
        log.error("Error: Must specify at least one eval task: [%s]"%args.eval_tasks)
        return 0 
      steps_log.append("Evaluating model on tasks: %s"%args.eval_tasks)
      
    if args.do_probe:
      try:
        assert args.probing_tasks
      except AssertionError:
        log.error("Error: Must specify at least one probing task: [%s]"%args.probing_tasks)
        return 0 
      try:
        assert args.probing_classifier
      except AssertionError:
        log.error("Error: Must specify a classifier for probing task: [%s]"%args.probing_classifier)
        return 0 
      steps_log.append("Probing with tasks %s using classifier %s"%(args.probing_tasks, args.probing_classifier))
    
    log.info("Will run the following steps:\n%s"%('\n'.join(steps_log)))

    if args.load_model_tmp:
      log.info("Loading existing model from %s..."%args.load_model_tmp)
      state_path = os.path.join(args.run_dir, args.load_model_tmp)
      load_model_state(model_mp, state_path, args.cuda)
    
    # Train on train tasks #
    if args.do_train:
        log.info("Training...")
        trainer, _, opt_params, schd_params = build_trainer(args, model,
                                                            args.max_vals)
        to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        stop_metric = train_tasks[0].val_metric if len(train_tasks) == 1 else 'macro_avg'
        best_epochs = trainer.train(train_tasks, stop_metric,
                                    args.val_interval, args.bpp_base,
                                    args.weighting_method, args.scaling_method,
                                    to_train, opt_params, schd_params,
                                    args.shared_optimizer, args.load_model)

    # Train just the task-specific components for eval tasks
    # TODO(Alex): currently will overwrite model checkpoints from training
    if args.train_for_eval:
        for task in eval_tasks:
                pred_module = getattr(model, "%s_mdl" % task.name)
                to_train = [(n, p) for n, p in pred_module.named_parameters() if p.requires_grad]
                trainer, _, opt_params, schd_params = build_trainer(args, model,
                                                                    args.eval_max_vals)
                best_epoch = trainer.train([task], task.val_metric,
                                           args.eval_val_interval, 1,
                                           args.weighting_method, args.scaling_method,
                                           to_train, opt_params, schd_params,
                                           args.shared_optimizer, args.load_model)

                best_epoch = best_epoch[task.name]
                layer_path = os.path.join(args.run_dir, "model_state_epoch_{}.th".format(best_epoch))
                load_model_state(model, layer_path, args.cuda)
    
    if args.do_eval:
        # Evaluate #
        log.info("Evaluating...")
        val_results, _ = evaluate(model, tasks, args.batch_size, args.cuda, "val")
        if args.write_preds:
            _, te_preds = evaluate(model, tasks, args.batch_size, args.cuda, "test")
            write_preds(te_preds, args.run_dir)
    
        write_results(val_results, os.path.join(args.exp_dir, "results.tsv"),
                      args.run_dir.split('/')[-1])
    
        log.info("Done!")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
