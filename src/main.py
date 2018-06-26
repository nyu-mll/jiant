'''Train a multi-task model using AllenNLP '''
import os
import sys
import time
import random
import argparse
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import ipdb as pdb
import torch

import config
from preprocess import build_tasks
from models import build_model
from trainer import build_trainer
from evaluate import evaluate, load_model_state, write_results, write_preds

import _pickle as pkl

def main(arguments):
    args = config.parse_arguments(arguments)

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
        except Exception:
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

    # Build model #
    log.info('Building model...')
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    log.info('\tFinished building model in %.3fs', time.time() - start_time)

    # Train on train tasks #
    if train_tasks and args.should_train:
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
    else:
        log.info("Skipping training.")
        best_epochs = {}

    # Select model checkpoint from training to load
    if args.force_load_epoch >= 0:  # force loading a particular epoch
        epoch_to_load = args.force_load_epoch
    elif "macro" in best_epochs:
        epoch_to_load = best_epochs['macro']
    else:
        serialization_files = os.listdir(args.run_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        if model_checkpoints:
            epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th"))
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
        if args.train_for_eval:
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
