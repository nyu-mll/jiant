'''Train a multi-task model using AllenNLP

To debug this, run with -m ipdb:

    python -m ipdb main.py --config_file ...
'''
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

from src.utils import config
from src.utils.utils import assert_for_log, maybe_make_dir, load_model_state, check_arg_name
from src.preprocess import build_tasks
from src.models import build_model
from src.trainer import build_trainer, build_trainer_params
from src import evaluate


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description='')
    # Configuration files
    parser.add_argument('--config_file', '-c', type=str, nargs="+",
                        help="Config file(s) (.conf) for model parameters.")
    parser.add_argument('--overrides', '-o', type=str, default=None,
                        help="Parameter overrides, as valid HOCON string.")

    parser.add_argument('--remote_log', '-r', action="store_true",
                        help="If true, enable remote logging on GCP.")

    parser.add_argument('--notify', type=str, default="",
                        help="Email address for job notifications.")

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


# Global notification handler, can be accessed outside main() during exception
# handling.
EMAIL_NOTIFIER = None


def main(cl_arguments):
    ''' Train or load a model. Evaluate on some tasks. '''
    cl_args = handle_arguments(cl_arguments)
    args = config.params_from_file(cl_args.config_file, cl_args.overrides)

    # Raise error if obsolete arg names are present
    check_arg_name(args)

    # Logistics #
    maybe_make_dir(args.project_dir)  # e.g. /nfs/jsalt/exp/$HOSTNAME
    maybe_make_dir(args.exp_dir)      # e.g. <project_dir>/jiant-demo
    maybe_make_dir(args.run_dir)      # e.g. <project_dir>/jiant-demo/sst
    log.getLogger().addHandler(log.FileHandler(args.local_log_path))

    if cl_args.remote_log:
        from src.utils import gcp
        gcp.configure_remote_logging(args.remote_log_name)

    if cl_args.notify:
        from src import emails
        global EMAIL_NOTIFIER
        log.info("Registering email notifier for %s", cl_args.notify)
        EMAIL_NOTIFIER = emails.get_notifier(cl_args.notify, args)

    if EMAIL_NOTIFIER:
        EMAIL_NOTIFIER(body="Starting run.", prefix="")

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

    # Prepare data #
    log.info("Loading tasks...")
    start_time = time.time()
    pretrain_tasks, target_tasks, vocab, word_embs = build_tasks(args)
    if any([t.val_metric_decreases for t in pretrain_tasks]) and any(
            [not t.val_metric_decreases for t in pretrain_tasks]):
        log.warn("\tMixing training tasks with increasing and decreasing val metrics!")
    tasks = sorted(set(pretrain_tasks + target_tasks), key=lambda x: x.name)
    log.info('\tFinished loading tasks in %.3fs', time.time() - start_time)
    log.info('\t Tasks: {}'.format([task.name for task in tasks]))

    # Build or load model #
    log.info('Building model...')
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    log.info('\tFinished building model in %.3fs', time.time() - start_time)

    # Check that necessary parameters are set for each step. Exit with error if not.
    steps_log = []

    assert_for_log(os.path.exists(args.load_eval_checkpoint),
                   "Error: Attempting to load model from non-existent path: [%s]" %
                   args.load_eval_checkpoint)
    assert_for_log(not args.do_pretrain,
            "Error: Attempting to train a model and then replace that model with one from a checkpoint.")
    steps_log.append("Loading model from path: %s" % args.load_eval_checkpoint)
    assert_for_log(args.target_tasks != "none",
                   "Error: Must specify at least one eval task: [%s]" % args.target_tasks)
    steps_log.append("Evaluating model on tasks: %s" % args.target_tasks)
    log.info("Will run the following steps:\n%s", '\n'.join(steps_log))

    # Select model checkpoint from main training run to load
    log.info("Loading existing model from %s...", args.load_eval_checkpoint)
    load_model_state(model, args.load_eval_checkpoint,
                     #args.cuda, task_names_to_avoid_loading, strict=strict)
                     args.cuda, task_names_to_avoid_loading=[], strict=False)


    # Evaluate #
    log.info("Evaluating...")
    splits_to_write = evaluate.parse_write_preds_arg(args.write_preds)
    if 'val' in splits_to_write:
        _, val_preds = evaluate.encode(model, target_tasks, args.batch_size,
                                       args.cuda, "val")
        evaluate.write_encs(target_tasks, val_preds, args.run_dir, 'val',
                            strict_glue_format=args.write_strict_glue_format)
    if 'test' in splits_to_write:
        _, te_preds = evaluate.encode(model, target_tasks,
                                      args.batch_size, args.cuda, "test")
        evaluate.write_encs(tasks, te_preds, args.run_dir, 'test',
                            strict_glue_format=args.write_strict_glue_format)

    log.info("Done!")


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
        if EMAIL_NOTIFIER is not None:
            EMAIL_NOTIFIER(body="Run completed successfully!", prefix="")
    except BaseException as e:
        # Make sure we log the trace for any crashes before exiting.
        log.exception("Fatal error in main():")
        if EMAIL_NOTIFIER is not None:
            import traceback
            tb_lines = traceback.format_exception(*sys.exc_info())
            EMAIL_NOTIFIER(body="".join(tb_lines), prefix="FAILED")
        raise e  # re-raise exception, in case debugger is attached.
        sys.exit(1)
    sys.exit(0)
