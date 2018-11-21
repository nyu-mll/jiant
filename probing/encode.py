'''Encode some sentences using a model '''
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

from src.utils import assert_for_log, maybe_make_dir, load_model_state, check_arg_name
from src.preprocess import build_tasks
from src.models import build_model
from src.trainer import build_trainer, build_trainer_params
from src.tasks import NLITypeProbingTask
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
    tasks = sorted(set(pretrain_tasks + target_tasks), key=lambda x: x.name)
    log.info('\tFinished loading tasks in %.3fs', time.time() - start_time)
    log.info('\t Tasks: {}'.format([task.name for task in tasks]))


    # Build or load model #
    log.info('Building model...')
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    log.info('\tFinished building model in %.3fs', time.time() - start_time)

    # Load a model #

    # Encode the sentences #
    model.encode(tasks)

    # Write the encodings to disk #

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
