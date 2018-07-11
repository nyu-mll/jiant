'''Train a multi-task model using AllenNLP '''
# pylint: disable=no-member
import argparse
import glob
import os
import subprocess
import random
import sys
import time

import logging as log
log.basicConfig(format='%(asctime)s: %(message)s',
                datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import torch

from src import config
from src import gcp

from src.utils import assert_for_log, maybe_make_dir, load_model_state
from src.preprocess import build_tasks
from src.models import build_model
from src.trainer import build_trainer, build_trainer_params
from src.evaluate import evaluate, write_results, write_preds
from src.tasks import NLITypeProbingTask


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description='')
    # Configuration files
    parser.add_argument('--config_file', '-c', type=str, nargs="+",
                        help="Config file(s) (.conf) for model parameters.")
    parser.add_argument('--overrides', '-o', type=str, default=None,
                        help="Parameter overrides, as valid HOCON string.")

    parser.add_argument('--remote_log', '-r', action="store_true",
                        help="If true, enable remote logging on GCP.")

    parser.add_argument('--tensorboard', '-t', action="store_true",
                        help="If true, will run Tensorboard server in a "
                        "subprocess, serving on the port given by "
                        "--tensorboard_port.")
    parser.add_argument('--tensorboard_port', type=int, default=6006)

    return parser.parse_args(cl_arguments)


def _try_logging_git_info():
    try:
        log.info("Waiting on git info....")
        c = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_branch_name = c.stdout.decode().strip()
        log.info("Git branch: %s", git_branch_name)
        c = subprocess.run(["git", "rev-parse", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_sha = c.stdout.decode().strip()
        log.info("Git SHA: %s", git_sha)
    except subprocess.TimeoutExpired as e:
        log.exception(e)
        log.warn("Git info not found. Moving right along...")


def _run_background_tensorboard(logdir, port):
    """Run a TensorBoard server in the background."""
    import atexit
    tb_args = ["tensorboard", "--logdir", logdir,
               "--port", str(port)]
    log.info("Starting TensorBoard server on port %d ...", port)
    tb_process = subprocess.Popen(tb_args)
    log.info("TensorBoard process: %d", tb_process.pid)

    def _kill_tb_child():
        log.info("Shutting down TensorBoard server on port %d ...", port)
        tb_process.terminate()
    atexit.register(_kill_tb_child)


def main(cl_arguments):
    ''' Train or load a model. Evaluate on some tasks. '''
    cl_args = handle_arguments(cl_arguments)
    args = config.params_from_file(cl_args.config_file, cl_args.overrides)

    # Logistics #
    maybe_make_dir(args.project_dir)  # e.g. /nfs/jsalt/exp/$HOSTNAME
    maybe_make_dir(args.exp_dir)      # e.g. <project_dir>/jiant-demo
    maybe_make_dir(args.run_dir)      # e.g. <project_dir>/jiant-demo/sst
    local_log_path = os.path.join(args.run_dir, args.log_file)
    log.getLogger().addHandler(log.FileHandler(local_log_path))

    if cl_args.remote_log:
        gcp.configure_remote_logging(args.remote_log_name)

    _try_logging_git_info()

    log.info("Parsed args: \n%s", args)

    config_file = os.path.join(args.run_dir, "params.conf")
    config.write_params(args, config_file)
    log.info("Saved config to %s", config_file)

    seed = random.randint(1, 10000) if args.random_seed < 0 else args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    log.info("Using random seed %d", seed)
    if args.cuda >= 0:
        try:
            if not torch.cuda.is_available():
                raise EnvironmentError("CUDA is not available, or not detected"
                                       " by PyTorch.")
            log.info("Using GPU %d", args.cuda)
            torch.cuda.set_device(args.cuda)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            log.warning(
                "GPU access failed. You might be using a CPU-only installation of PyTorch. Falling back to CPU.")
            args.cuda = -1
    if args.debug:
        log.info("Debugging is ON; we're importing pdb!")
        try:
            pass
            #import ipdb as pdb
        except ImportError:
            pass
            #import pdb

    # Prepare data #
    log.info("Loading tasks...")
    start_time = time.time()
    train_tasks, eval_tasks, vocab, word_embs = build_tasks(args)
    assert_for_log(not (len(train_tasks) > 1 and
                        any(train_task.val_metric_decreases for train_task in train_tasks)),
                   "Attempting multitask training with a mix of increasing and decreasing metrics. "
                   "This is not currently supported. (We haven't set it up yet.)")

    tasks = train_tasks + eval_tasks
    log.info('\tFinished loading tasks in %.3fs', time.time() - start_time)

    # Build or load model #
    log.info('Building model...')
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    log.info('\tFinished building model in %.3fs', time.time() - start_time)

    # Check that necessary parameters are set for each step. Exit with error if not.
    steps_log = []

    if not args.load_eval_checkpoint == 'none':
        assert_for_log(os.path.exists(args.load_eval_checkpoint),
                       "Error: Attempting to load model from non-existent path: [%s]" %
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

    # Start Tensorboard if requested
    if cl_args.tensorboard:
        tb_logdir = os.path.join(args.run_dir, "tensorboard")
        _run_background_tensorboard(tb_logdir, cl_args.tensorboard_port)

    log.info("Will run the following steps:\n%s", '\n'.join(steps_log))
    if args.do_train:
        # Train on train tasks #
        log.info("Training...")
        params = build_trainer_params(args, 'none', args.max_vals, args.val_interval)
        stop_metric = train_tasks[0].val_metric if len(train_tasks) == 1 else 'macro_avg'
        should_decrease = train_tasks[0].val_metric_decreases if len(train_tasks) == 1 else False
        trainer, _, opt_params, schd_params = build_trainer(params, model,
                                                            args.run_dir,
                                                            should_decrease)
        to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        best_epochs = trainer.train(train_tasks, stop_metric,
                                    args.batch_size, args.bpp_base,
                                    args.weighting_method, args.scaling_method,
                                    to_train, opt_params, schd_params,
                                    args.shared_optimizer, args.load_model, phase="main")

    # Select model checkpoint from main training run to load
    # is not None and args.load_eval_checkpoint != "none":
    if not args.load_eval_checkpoint == "none":
        log.info("Loading existing model from %s...", args.load_eval_checkpoint)
        load_model_state(model, args.load_eval_checkpoint, args.cuda, args.skip_task_models)
    else:
        # Look for eval checkpoints (available only if we're restoring from a run that already
        # finished), then look for training checkpoints.
        eval_best = glob.glob(os.path.join(args.run_dir,
                                           "model_state_eval_best.th"))
        if len(eval_best) > 0:
            load_model_state(model, eval_best[0], args.cuda, args.skip_task_models)
        else:
            macro_best = glob.glob(os.path.join(args.run_dir,
                                                "model_state_main_epoch_*.best_macro.th"))
            if len(macro_best) > 0:
                assert_for_log(len(macro_best) == 1, "Too many best checkpoints. Something is wrong.")
                load_model_state(model, macro_best[0], args.cuda, args.skip_task_models)
            else:
                assert_for_log(
                    args.allow_untrained_encoder_parameters,
                    "No best checkpoint found to evaluate.")
                log.warning("Evaluating untrained encoder parameters!")

    # Train just the task-specific components for eval tasks.
    if args.train_for_eval:
        for task in eval_tasks:
            pred_module = getattr(model, "%s_mdl" % task.name)
            to_train = [(n, p) for n, p in pred_module.named_parameters() if p.requires_grad]
            params = build_trainer_params(args, task.name, args.eval_max_vals,
                                          args.eval_val_interval)
            trainer, _, opt_params, schd_params = build_trainer(params, model,
                                                                args.run_dir,
                                                                task.val_metric_decreases)
            best_epoch = trainer.train([task], task.val_metric,
                                       args.batch_size, 1,
                                       args.weighting_method, args.scaling_method,
                                       to_train, opt_params, schd_params,
                                       args.shared_optimizer, load_model=False, phase="eval")

            # The best checkpoint will accumulate the best parameters for each task.
            # This logic looks strange. We think it works.
            best_epoch = best_epoch[task.name]
            layer_path = os.path.join(args.run_dir, "model_state_eval_best.th")
            load_model_state(model, layer_path, args.cuda, skip_task_models=False)

    if args.do_eval:
        # Evaluate #
        log.info("Evaluating...")
        val_results, _ = evaluate(model, tasks, args.batch_size, args.cuda, "val")
        if args.write_preds:
            if isinstance(tasks[0], NLITypeProbingTask):
                _, te_preds = evaluate(model, tasks, args.batch_size, args.cuda, "val")
            else:
                _, te_preds = evaluate(model, tasks, args.batch_size, args.cuda, "test")
            write_preds(te_preds, args.run_dir)

        write_results(val_results, os.path.join(args.exp_dir, "results.tsv"),
                      args.run_dir.split('/')[-1])

    log.info("Done!")


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except BaseException:
        # Make sure we log the trace for any crashes before exiting.
        log.exception("Fatal error in main():")
        sys.exit(1)
    sys.exit(0)
