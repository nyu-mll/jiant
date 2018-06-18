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

from allennlp.data.iterators import BasicIterator, BucketIterator
from utils import device_mapping

from preprocess import build_tasks
from models import build_model
from trainer import build_trainer
from evaluate import evaluate

def main(arguments):
    ''' Train or load a model. Evaluate on some tasks. '''
    parser = argparse.ArgumentParser(description='')

    # Logistics
    parser.add_argument('--cuda', help='-1 if no CUDA, else gpu id', type=int, default=0)
    parser.add_argument('--random_seed', help='random seed to use', type=int, default=19)

    # Paths and logging
    parser.add_argument('--log_file', help='file to log to', type=str, default='log.log')
    parser.add_argument('--exp_dir', help='directory containing shared preprocessing', type=str)
    parser.add_argument('--run_dir', help='directory for saving results, models, etc.', type=str)
    parser.add_argument('--word_embs_file', help='file containing word embs', type=str, default='')
    parser.add_argument('--preproc_file', help='file containing saved preprocessing stuff',
                        type=str, default='preproc.pkl')

    # Time saving flags
    parser.add_argument('--should_train', help='1 if should train model', type=int, default=1)
    parser.add_argument('--load_model', help='1 if load from checkpoint', type=int, default=1)
    parser.add_argument('--load_epoch', help='Force loading from a certain epoch', type=int,
                        default=-1)
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
    parser.add_argument('--dropout_embs', help='dropout rate for embeddings', type=float, default=.2)
    parser.add_argument('--d_word', help='dimension of word embeddings', type=int, default=300)
    parser.add_argument('--glove', help='1 if use glove, else from scratch', type=int, default=1)
    parser.add_argument('--train_words', help='1 if make word embs trainable', type=int, default=0)
    parser.add_argument('--elmo', help='1 if use elmo', type=int, default=0)
    parser.add_argument('--deep_elmo', help='1 if use elmo post LSTM', type=int, default=0)
    parser.add_argument('--elmo_no_glove', help='1 if no glove, assuming elmo', type=int, default=0)
    parser.add_argument('--cove', help='1 if use cove', type=int, default=0)

    # Model options
    parser.add_argument('--sent_enc', help='type of sent encoder to use', type=str, default='rnn',
                        choices=['bow', 'rnn'])
    parser.add_argument('--pair_enc', help='type of pair encoder to use', type=str, default='simple',
                        choices=['simple', 'attn'])
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

    args = parser.parse_args(arguments)

    # Logistics #
    log.basicConfig(format='%(asctime)s: %(message)s', level=log.INFO, datefmt='%m/%d %I:%M:%S %p')
    log_file = os.path.join(args.run_dir, args.log_file)
    file_handler = log.FileHandler(log_file)
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
    # TODO(Alex): move iterator creation
    iterator = BasicIterator(args.batch_size)
    #iterator = BucketIterator(sorting_keys=[("sentence1", "num_tokens")], batch_size=args.batch_size)
    trainer, train_params, opt_params, schd_params = build_trainer(args, args.trainer_type, model, iterator)

    # Train #
    if train_tasks and args.should_train:
        #to_train = [p for p in model.parameters() if p.requires_grad]
        to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if args.trainer_type == 'mtl':
            best_epochs = trainer.train(train_tasks, args.task_ordering, args.val_interval,
                                        args.max_vals, args.bpp_method, args.bpp_base, to_train,
                                        opt_params, schd_params, args.load_model)
        elif args.trainer_type == 'sampling':
            if args.weighting_method == 'uniform':
                log.info("Sampling tasks uniformly")
            elif args.weighting_method == 'proportional':
                log.info("Sampling tasks proportional to number of training batches")

            if args.scaling_method == 'max':
                # divide by # batches, multiply by max # batches
                log.info("Scaling losses to largest task")
            elif args.scaling_method == 'min':
                # divide by # batches, multiply by fewest # batches
                log.info("Scaling losses to the smallest task")
            elif args.scaling_method == 'unit':
                log.info("Dividing losses by number of training batches")
            best_epochs = trainer.train(train_tasks, args.val_interval, args.bpp_base,
                                        args.weighting_method, args.scaling_method, to_train,
                                        opt_params, schd_params, args.shared_optimizer,
                                        args.load_model)
    else:
        log.info("Skipping training.")
        best_epochs = {}

    # train just the classifiers for eval tasks
    for task in eval_tasks:
        pred_layer = getattr(model, "%s_pred_layer" % task.name)
        to_train = pred_layer.parameters()
        #trainer = MultiTaskTrainer.from_params(model, args.run_dir + '/%s/' % task.name,
        #                                       iterator, copy.deepcopy(train_params))
        trainer = None # todo
        trainer.train([task], args.task_ordering, 1, args.max_vals, 'percent_tr', 1, to_train,
                      opt_params, schd_params, 1)
        layer_path = os.path.join(args.run_dir, task.name, "%s_best.th" % task.name)
        layer_state = torch.load(layer_path, map_location=device_mapping(args.cuda))
        model.load_state_dict(layer_state)

    # Evaluate: load the different task best models and evaluate them
    # TODO(Alex): put this in evaluate file
    all_results = {}

    if not best_epochs and args.load_epoch >= 0:
        epoch_to_load = args.load_epoch
    elif not best_epochs and not args.load_epoch:
        serialization_files = os.listdir(args.run_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th")) \
                             for x in model_checkpoints])
    else:
        epoch_to_load = -1

    #for task in [task.name for task in train_tasks] + ['micro', 'macro']:
    for task in ['macro']:
        log.info("Testing on %s..." % task)

        # Load best model
        load_idx = best_epochs[task] if best_epochs else epoch_to_load
        model_path = os.path.join(args.run_dir, "model_state_epoch_{}.th".format(load_idx))
        model_state = torch.load(model_path, map_location=device_mapping(args.cuda))
        model.load_state_dict(model_state)

        # Test evaluation and prediction
        # could just filter out tasks to get what i want...
        #tasks = [task for task in tasks if 'mnli' in task.name]
        te_results, te_preds = evaluate(model, tasks, iterator, cuda_device=args.cuda, split="test")
        val_results, _ = evaluate(model, tasks, iterator, cuda_device=args.cuda, split="val")

        if task == 'macro':
            all_results[task] = (val_results, te_results, model_path)
            for eval_task, task_preds in te_preds.items(): # write predictions for each task
                #if 'mnli' not in eval_task:
                #    continue
                idxs_and_preds = [(idx, pred) for pred, idx in zip(task_preds[0], task_preds[1])]
                idxs_and_preds.sort(key=lambda x: x[0])
                if 'mnli' in eval_task:
                    pred_map = {0: 'neutral', 1: 'entailment', 2: 'contradiction'}
                    with open(os.path.join(args.run_dir, "%s-m.tsv" % (eval_task)), 'w') as pred_fh:
                        pred_fh.write("index\tprediction\n")
                        split_idx = 0
                        for idx, pred in idxs_and_preds[:9796]:
                            pred = pred_map[pred]
                            pred_fh.write("%d\t%s\n" % (split_idx, pred))
                            split_idx += 1
                    with open(os.path.join(args.run_dir, "%s-mm.tsv" % (eval_task)), 'w') as pred_fh:
                        pred_fh.write("index\tprediction\n")
                        split_idx = 0
                        for idx, pred in idxs_and_preds[9796:9796+9847]:
                            pred = pred_map[pred]
                            pred_fh.write("%d\t%s\n" % (split_idx, pred))
                            split_idx += 1
                    with open(os.path.join(args.run_dir, "diagnostic.tsv"), 'w') as pred_fh:
                        pred_fh.write("index\tprediction\n")
                        split_idx = 0
                        for idx, pred in idxs_and_preds[9796+9847:]:
                            pred = pred_map[pred]
                            pred_fh.write("%d\t%s\n" % (split_idx, pred))
                            split_idx += 1
                else:
                    with open(os.path.join(args.run_dir, "%s.tsv" % (eval_task)), 'w') as pred_fh:
                        pred_fh.write("index\tprediction\n")
                        for idx, pred in idxs_and_preds:
                            if 'sts-b' in eval_task:
                                pred_fh.write("%d\t%.3f\n" % (idx, pred))
                            elif 'rte' in eval_task:
                                pred = 'entailment' if pred else 'not_entailment'
                                pred_fh.write('%d\t%s\n' % (idx, pred))
                            elif 'squad' in eval_task:
                                pred = 'entailment' if pred else 'not_entailment'
                                pred_fh.write('%d\t%s\n' % (idx, pred))
                            else:
                                pred_fh.write("%d\t%d\n" % (idx, pred))

            with open(os.path.join(args.exp_dir, "results.tsv"), 'a') as results_fh: # aggregate results easily
                run_name = args.run_dir.split('/')[-1]
                all_metrics_str = ', '.join(['%s: %.3f' % (metric, score) for \
                                            metric, score in val_results.items()])
                results_fh.write("%s\t%s\n" % (run_name, all_metrics_str))
    log.info("Done testing")

    # Dump everything to a pickle for posterity
    pkl.dump(all_results, open(os.path.join(args.run_dir, "results.pkl"), 'wb'))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
