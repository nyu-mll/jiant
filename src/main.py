'''Train a multi-task model using AllenNLP '''
import argparse
import glob
import json
import os
import random
import sys
import time

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import torch

import config
from preprocess import build_tasks
from models import build_model
from trainer import build_trainer
from evaluate import evaluate, load_model_state, write_results, write_preds
from utils import assert_for_log

import _pickle as pkl

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
JIANT_BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
DEFAULT_CONFIG_FILE = os.path.join(JIANT_BASE_DIR, "config/defaults.conf")

def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description='')
    # Configuration files
    parser.add_argument('--config_file',
                        help="Config file (.conf) for model parameters.",
                        type=str, default=DEFAULT_CONFIG_FILE)
    parser.add_argument('--overrides', help="Parameter overrides, as valid HOCON string.", type=str, default=None)

    return parser.parse_args(cl_arguments)


def main(cl_arguments):
    ''' Train or load a model. Evaluate on some tasks. '''
    args = handle_arguments(cl_arguments)
    args = config.params_from_file(args.config_file, args.overrides)

    # Logistics #
    if not os.path.isdir(args.exp_dir):
        os.mkdir(args.exp_dir)
    if not os.path.isdir(args.run_dir):
        os.mkdir(args.run_dir)
    log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir,
                                                            args.log_file)))
    log.info("Parsed args: \n%s", args)

    config_file = os.path.join(args.run_dir, "params.conf")
    config.write_params(args, config_file)
    log.info("Saved config to %s", config_file)

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

    # Build or load model #
    log.info('Building model...')
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    log.info('\tFinished building model in %.3fs', time.time() - start_time)

    # Check that necessary parameters are set for each step. Exit with error if not.
    steps_log = []

    if not(args.load_eval_checkpoint == 'none'):
        assert_for_log(os.path.exists(args.load_eval_checkpoint),
                "Error: Attempting to load model from non-existent path: [%s]" % \
                args.load_eval_checkpoint)
        steps_log.append("Loading model from path: %s" % args.load_eval_checkpoint)

    if args.do_train:
        assert_for_log(args.train_tasks != "none", 
            "Error: Must specify at least on training task: [%s]" % args.train_tasks)
        steps_log.append("Training model on tasks: %s" % args.train_tasks)

    if args.train_for_eval:
        steps_log.append("Re-training model for individual eval tasks")

    if args.do_eval:
        assert_for_log(args.eval_tasks != "none", 
            "Error: Must specify at least one eval task: [%s]" % args.eval_tasks)
        steps_log.append("Evaluating model on tasks: %s" % args.eval_tasks)

    log.info("Will run the following steps:\n%s" % ('\n'.join(steps_log)))
    if args.do_train:
        # Train on train tasks #
        log.info("Training...")
        trainer, _, opt_params, schd_params = build_trainer(args, model,
                                                            args.max_vals)
        to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        stop_metric = train_tasks[0].val_metric if len(train_tasks) == 1 else 'macro_avg'
        best_epochs = trainer.train(train_tasks, stop_metric,
                                    args.val_interval, args.bpp_base,
                                    args.weighting_method, args.scaling_method,
                                    to_train, opt_params, schd_params,
                                    args.shared_optimizer, args.load_model, phase="main")

    # Select model checkpoint from main training run to load
    # is not None and args.load_eval_checkpoint != "none":
    if not(args.load_eval_checkpoint == "none"):
        log.info("Loading existing model from %s..." % args.load_eval_checkpoint)
        load_model_state(model, args.load_eval_checkpoint, args.cuda)
    else:
        macro_best = glob.glob(os.path.join(args.run_dir,
                                            "model_state_main_epoch_*.best_macro.th"))
        assert_for_log(len(macro_best) > 0, "No best checkpoint found to evaluate.")
        assert_for_log(len(macro_best) == 1, "Too many best checkpoints. Something is wrong.")
        load_model_state(model, macro_best[0], args.cuda)

    # Train just the task-specific components for eval tasks.
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
                                       args.shared_optimizer, load_model=False, phase="eval")

            # The best checkpoint will accumulate the best parameters for each task.
            # This logic looks strange. We think it works.
            best_epoch = best_epoch[task.name]
            layer_path = os.path.join(args.run_dir, "model_state_eval_best.th")
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
