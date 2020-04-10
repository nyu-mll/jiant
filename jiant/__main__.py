"""Main flow for jiant.

To debug this, run with -m ipdb:

    ipdb3 jiant/__main__.py  --config_file ...
"""
# pylint: disable=no-member
import logging as log
from typing import Iterable

log.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
)  # noqa
import argparse
import glob
import io
import os
from pkg_resources import resource_filename
import random
import subprocess
import sys
import time
import copy
import torch
import torch.nn as nn

from jiant import evaluate
from jiant.models import build_model
from jiant.preprocess import build_tasks
from jiant import tasks as task_modules
from jiant.trainer import build_trainer
from jiant.utils import config, tokenizers
from jiant.utils.options import parse_cuda_list_arg
from jiant.utils.utils import (
    assert_for_log,
    load_model_state,
    maybe_make_dir,
    parse_json_diff,
    sort_param_recursive,
    select_relevant_print_args,
    check_for_previous_checkpoints,
    select_pool_type,
    delete_all_checkpoints,
    get_model_attribute,
    uses_cuda,
)


# Global notification handler, can be accessed outside main() during exception handling.
EMAIL_NOTIFIER = None


def handle_arguments(cl_arguments: Iterable[str]) -> argparse.Namespace:
    """Defines jiant's CLI argument parsing logic

    Parameters
    ----------
    cl_arguments : Iterable[str]
        An sys.argv-style args obj.

    Returns
    -------
    argparse.Namespace
        A map of params and parsed args
    """
    parser = argparse.ArgumentParser(description="")
    # Configuration files
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        nargs="+",
        default=resource_filename("jiant", "config/defaults.conf"),
        help="Config file(s) (.conf) for model parameters.",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        type=str,
        default=None,
        help="Parameter overrides, as valid HOCON string.",
    )

    parser.add_argument(
        "--remote_log", "-r", action="store_true", help="If true, enable remote logging on GCP."
    )

    parser.add_argument(
        "--notify", type=str, default="", help="Email address for job notifications."
    )

    parser.add_argument(
        "--tensorboard",
        "-t",
        action="store_true",
        help="If true, will run Tensorboard server in a "
        "subprocess, serving on the port given by "
        "--tensorboard_port.",
    )
    parser.add_argument("--tensorboard_port", type=int, default=6006)

    return parser.parse_args(cl_arguments)


def setup_target_task_training(args, target_tasks, model, strict):
    """
    Gets the model path used to restore model after each target
    task run, and saves current state if no other previous checkpoint can
    be used as the model path.
    The logic for loading the correct model state for target task training is:
    1) If load_target_train_checkpoint is used, then load the weights from that checkpoint.
    2) If we did pretraining, then load the best model from pretraining.
    3) Default case: we save untrained encoder weights.

    Parameters
    ----------------
    args: Params object
    target_tasks: list of target Task objects
    mdoel: a MultiTaskModel object

    Returns
    ----------------
    model_path: str
    """
    model_path = get_best_checkpoint_path(args, "target_train")
    if model_path is None:
        # We want to do target training without pretraining, thus
        # we need to first create a checkpoint to come back to for each of
        # the target tasks to finetune.
        if args.transfer_paradigm == "frozen":
            assert_for_log(
                args.allow_untrained_encoder_parameters,
                "No best checkpoint found to target train on. Set \
                `allow_untrained_encoder_parameters` if you really want to use an untrained \
                encoder.",
            )
        model_path = os.path.join(args.run_dir, "model_state_untrained_pre_target_train.th")
        torch.save(model.state_dict(), model_path)

    return model_path


def check_configurations(args, pretrain_tasks, target_tasks):
    """
    Checks configurations for any obvious logical flaws
    and that necessary parameters are set for each step -
    throws asserts and exits if found.

    Parameters
    ----------------
    args: Params object
    pretrain_tasks: list of pretraining Task objects
    target_tasks: list of target task training Task objects

    Returns
    ----------------
    None
    """
    steps_log = io.StringIO()
    if any([t.val_metric_decreases for t in pretrain_tasks]) and any(
        [not t.val_metric_decreases for t in pretrain_tasks]
    ):
        log.warn("\tMixing training tasks with increasing and decreasing val metrics!")

    assert (
        hasattr(args, "accumulation_steps") and args.accumulation_steps >= 1
    ), "accumulation_steps must be a positive int."

    if args.load_target_train_checkpoint != "none":
        assert_for_log(
            not args.do_pretrain,
            "Error: Attempting to train a model and then replace that model with one from "
            "a checkpoint.",
        )
        steps_log.write("Loading model from path: %s \n" % args.load_target_train_checkpoint)

    assert_for_log(
        args.transfer_paradigm in ["finetune", "frozen"],
        "Transfer paradigm %s not supported!" % args.transfer_paradigm,
    )

    if args.do_pretrain:
        assert_for_log(
            args.pretrain_tasks not in ("none", "", None),
            "Error: Must specify at least one pretraining task: [%s]" % args.pretrain_tasks,
        )
        steps_log.write("Training model on tasks: %s \n" % args.pretrain_tasks)

    if args.do_target_task_training:
        assert_for_log(
            args.target_tasks not in ("none", "", None),
            "Error: Must specify at least one target task: [%s]" % args.target_tasks,
        )
        steps_log.write("Re-training model for individual target tasks \n")
        assert_for_log(
            len(set(pretrain_tasks).intersection(target_tasks)) == 0
            or args.allow_reuse_of_pretraining_parameters
            or args.do_pretrain == 0,
            "If you're pretraining on a task you plan to reuse as a target task, set\n"
            "allow_reuse_of_pretraining_parameters = 1 (risky), or train in two steps:\n"
            "train with do_pretrain = 1, do_target_task_training = 0, stop, and restart with\n"
            "do_pretrain = 0 and do_target_task_training = 1.",
        )
    if args.do_full_eval:
        assert_for_log(
            args.target_tasks != "none",
            "Error: Must specify at least one target task: [%s]" % args.target_tasks,
        )
        if not args.do_target_task_training:
            untrained_tasks = set(
                config.get_task_attr(args, task.name, "use_classifier", default=task.name)
                for task in target_tasks
            )
            if args.do_pretrain:
                untrained_tasks -= set(
                    config.get_task_attr(args, task.name, "use_classifier", default=task.name)
                    for task in pretrain_tasks
                )
            if len(untrained_tasks) > 0:
                assert (
                    args.load_model
                    or args.load_target_train_checkpoint not in ["none", ""]
                    or args.allow_untrained_encoder_parameters
                ), f"Evaluating a target task model on tasks {untrained_tasks} "
                "without training it on this run or loading a checkpoint. "
                "Set `allow_untrained_encoder_parameters` if you really want to use "
                "an untrained task model."
                log.warning(
                    f"Evauluating a target task model on tasks {untrained_tasks} without training "
                    "it in this run. It's up to you to ensure that you are loading parameters "
                    "that were sufficiently trained for this task."
                )
        steps_log.write("Evaluating model on tasks: %s \n" % args.target_tasks)

    log.info("Will run the following steps for this experiment:\n%s", steps_log.getvalue())
    steps_log.close()


def _log_git_info():
    try:
        # Make sure we run git in the directory that contains this file, even if the working
        # directory is elsewhere.
        main_dir = os.path.dirname(os.path.abspath(__file__))

        # Use git to get branch/commit ID information.
        c = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            timeout=10,
            stdout=subprocess.PIPE,
            cwd=main_dir,
        )
        git_branch_name = c.stdout.decode().strip()
        log.info("Git branch: %s", git_branch_name)
        c = subprocess.run(
            ["git", "rev-parse", "HEAD"], timeout=10, stdout=subprocess.PIPE, cwd=main_dir
        )
        git_sha = c.stdout.decode().strip()
        log.info("Git SHA: %s", git_sha)
    except subprocess.TimeoutExpired as e:
        log.exception(e)
        log.warn("Git info not found. Moving right along...")


def _run_background_tensorboard(logdir, port):
    """Run a TensorBoard server in the background."""
    import atexit

    tb_args = ["tensorboard", "--logdir", logdir, "--port", str(port)]
    log.info("Starting TensorBoard server on port %d ...", port)
    tb_process = subprocess.Popen(tb_args)
    log.info("TensorBoard process: %d", tb_process.pid)

    def _kill_tb_child():
        log.info("Shutting down TensorBoard server on port %d ...", port)
        tb_process.terminate()

    atexit.register(_kill_tb_child)


def get_best_checkpoint_path(args, phase, task_name=None):
    """ Look in run_dir for model checkpoint to load when setting up for
    phase = target_train or phase = eval.
    Hierarchy is:
        If phase == target_train:
            1) user-specified target task checkpoint
            2) best task-specific checkpoint from pretraining stage
        If phase == eval:
            1) user-specified eval checkpoint
            2) best task-specific checkpoint for target_train, used when evaluating
            3) best pretraining checkpoint
    If all these fail, then we default to None.
    """
    checkpoint = []
    if phase == "target_train":
        if args.load_target_train_checkpoint not in ("none", ""):
            checkpoint = glob.glob(args.load_target_train_checkpoint)
            assert len(checkpoint) > 0, (
                "Specified load_target_train_checkpoint not found: %r"
                % args.load_target_train_checkpoint
            )
        else:
            checkpoint = glob.glob(os.path.join(args.run_dir, "model_state_pretrain_val_*.best.th"))
    if phase == "eval":
        if args.load_eval_checkpoint not in ("none", ""):
            checkpoint = glob.glob(args.load_eval_checkpoint)
            assert len(checkpoint) > 0, (
                "Specified load_eval_checkpoint not found: %r" % args.load_eval_checkpoint
            )
        else:
            # Get the best checkpoint from the target_train phase to evaluate on.
            assert task_name is not None, "Specify a task checkpoint to evaluate from."
            checkpoint = glob.glob(
                os.path.join(args.run_dir, task_name, "model_state_target_train_val_*.best.th")
            )
            if len(checkpoint) == 0:
                checkpoint = glob.glob(
                    os.path.join(args.run_dir, "model_state_pretrain_val_*.best.th")
                )

    if len(checkpoint) > 0:
        assert_for_log(len(checkpoint) == 1, "Too many best checkpoints. Something is wrong.")
        return checkpoint[0]
    return None


def evaluate_and_write(args, model, tasks, splits_to_write, cuda_device):
    """ Evaluate a model on dev and/or test, then write predictions """
    val_results, val_preds = evaluate.evaluate(model, tasks, args.batch_size, cuda_device, "val")
    if "val" in splits_to_write:
        evaluate.write_preds(
            tasks, val_preds, args.run_dir, "val", strict_glue_format=args.write_strict_glue_format
        )
    if "test" in splits_to_write:
        _, te_preds = evaluate.evaluate(model, tasks, args.batch_size, cuda_device, "test")
        evaluate.write_preds(
            tasks, te_preds, args.run_dir, "test", strict_glue_format=args.write_strict_glue_format
        )

    run_name = args.get("run_name", os.path.basename(args.run_dir))
    results_tsv = os.path.join(args.exp_dir, "results.tsv")
    log.info("Writing results for split 'val' to %s", results_tsv)
    evaluate.write_results(val_results, results_tsv, run_name=run_name)


def initial_setup(args: config.Params, cl_args: argparse.Namespace) -> (config.Params, int):
    """Perform setup steps:

    1. create project, exp, and run dirs if they don't already exist
    2. create log formatter
    3. configure GCP remote logging
    4. set up email notifier
    5. log git info
    6. write the config out to file
    7. log diff between default and experiment's configs
    8. choose torch's and random's random seed
    9. if config specifies a single GPU, then set the GPU's random seed (doesn't cover multi-GPU)
    10. resolve "auto" settings for tokenizer and pool_type parameters

    Parameters
    ----------
    args : config.Params
        config map
    cl_args : argparse.Namespace
        mapping named arguments to parsed values

    Returns
    -------
    args : config.Params
        config map
    seed : int
        random's and pytorch's random seed

    """
    output = io.StringIO()
    maybe_make_dir(args.project_dir)  # e.g. /nfs/jsalt/exp/$HOSTNAME
    maybe_make_dir(args.exp_dir)  # e.g. <project_dir>/jiant-demo
    maybe_make_dir(args.run_dir)  # e.g. <project_dir>/jiant-demo/sst
    log_fh = log.FileHandler(args.local_log_path)
    log_fmt = log.Formatter("%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p")
    log_fh.setFormatter(log_fmt)
    log.getLogger().addHandler(log_fh)

    if cl_args.remote_log:
        from jiant.utils import gcp

        gcp.configure_remote_logging(args.remote_log_name)

    if cl_args.notify:
        from jiant.utils import emails

        global EMAIL_NOTIFIER
        log.info("Registering email notifier for %s", cl_args.notify)
        EMAIL_NOTIFIER = emails.get_notifier(cl_args.notify, args)

    if EMAIL_NOTIFIER:
        EMAIL_NOTIFIER(body="Starting run.", prefix="")

    _log_git_info()
    config_file = os.path.join(args.run_dir, "params.conf")
    config.write_params(args, config_file)

    print_args = select_relevant_print_args(args)
    log.info("Parsed args: \n%s", print_args)

    log.info("Saved config to %s", config_file)

    seed = random.randint(1, 10000) if args.random_seed < 0 else args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    log.info("Using random seed %d", seed)
    if isinstance(args.cuda, int) and args.cuda >= 0:
        # If only running on one GPU.
        try:
            if not torch.cuda.is_available():
                raise EnvironmentError("CUDA is not available, or not detected" " by PyTorch.")
            log.info("Using GPU %d", args.cuda)
            torch.cuda.set_device(args.cuda)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            log.warning(
                "GPU access failed. You might be using a CPU-only installation of PyTorch. "
                "Falling back to CPU."
            )
            args.cuda = -1

    if args.tokenizer == "auto":
        args.tokenizer = tokenizers.select_tokenizer(args)
    if args.pool_type == "auto":
        args.pool_type = select_pool_type(args)

    return args, seed


def check_arg_name(args: config.Params):
    """Check for obsolete params in config, throw exceptions if obsolete params are found.

    Parameters
    ----------
    args: config.Params
        config map

    Raises
    ------
    AssertionError
        If obsolete parameter names are present in config

    """
    # Mapping - key: old name, value: new name
    name_dict = {
        "task_patience": "lr_patience",
        "do_train": "do_pretrain",
        "train_for_eval": "do_target_task_training",
        "do_eval": "do_full_eval",
        "train_tasks": "pretrain_tasks",
        "eval_tasks": "target_tasks",
        "eval_data_fraction": "target_train_data_fraction",
        "eval_val_interval": "target_train_val_interval",
        "eval_max_vals": "target_train_max_vals",
        "eval_data_fraction": "target_train_data_fraction",
    }
    for task in task_modules.ALL_GLUE_TASKS + task_modules.ALL_SUPERGLUE_TASKS:
        assert_for_log(
            not args.regex_contains("^{}_".format(task)),
            "Error: Attempting to load old task-specific args for task %s, please refer to the "
            "master branch's default configs for the most recent task specific argument "
            "structures." % task,
        )
    for old_name, new_name in name_dict.items():
        assert_for_log(
            old_name not in args,
            "Error: Attempting to load old arg name %s, please update to new name %s."
            % (old_name, name_dict[old_name]),
        )
    old_input_module_vals = [
        "elmo",
        "elmo_chars_only",
        "bert_model_name",
        "openai_transformer",
        "word_embs",
    ]
    for input_type in old_input_module_vals:
        assert_for_log(
            input_type not in args,
            "Error: Attempting to load old arg name %s, please use input_module config "
            "parameter and refer to master branch's default configs for current way to specify %s."
            % (input_type, input_type),
        )


def load_model_for_target_train_run(args, ckpt_path, model, strict, task, cuda_devices):
    """
        Function that reloads model if necessary and extracts trainable parts
        of the model in preparation for target_task training.
        It only reloads model after the first task is trained.

        Parameters
        -------------------
        args: config.Param object,
        ckpt_path: str: path to reload model from,
        model: MultiTaskModel object,
        strict: bool,
        task: Task object

        Returns
        -------------------
        to_train: List of tuples of (name, weight) of trainable parameters

    """
    load_model_state(model, ckpt_path, cuda_devices, skip_task_models=[task.name], strict=strict)
    if args.transfer_paradigm == "finetune":
        # Train both the task specific models as well as sentence encoder.
        to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    else:  # args.transfer_paradigm == "frozen":
        # will be empty if args.input_module != "elmo", scalar_mix_0 should always be
        # pretrain scalars
        elmo_scalars = [
            (n, p)
            for n, p in model.named_parameters()
            if "scalar_mix" in n and "scalar_mix_0" not in n
        ]
        # Fails when sep_embs_for_skip is 0 and elmo_scalars has nonzero
        # length.
        assert_for_log(
            not elmo_scalars or args.sep_embs_for_skip,
            "Error: ELMo scalars loaded and will be updated in do_target_task_training but "
            "they should not be updated! Check sep_embs_for_skip flag or make an issue.",
        )
        # Only train task-specific module

        pred_module = get_model_attribute(model, "%s_mdl" % task.name, cuda_devices)
        to_train = [(n, p) for n, p in pred_module.named_parameters() if p.requires_grad]
        to_train += elmo_scalars
    model = model.cuda() if uses_cuda(cuda_devices) else model
    if isinstance(cuda_devices, list):
        model = nn.DataParallel(model, device_ids=cuda_devices)
    return to_train


def get_pretrain_stop_metric(early_stopping_method, pretrain_tasks):
    """
    Get stop_metric, which is used for early stopping. 

    Parameters
    -------------------
     early_stopping_method: str,
     pretrain_tasks: List[Task]

    Returns 
    -------------------
    stop_metric: str

    """
    if early_stopping_method != "auto":
        pretrain_names = [task.name for task in pretrain_tasks]
        if early_stopping_method in pretrain_names:
            index = pretrain_names.index(early_stopping_method)
            stop_metric = pretrain_tasks[index].val_metric
        else:
            raise ValueError("args.early_stopping_method must be either 'auto' or a task name")

    else:
        stop_metric = pretrain_tasks[0].val_metric if len(pretrain_tasks) == 1 else "macro_avg"
    return stop_metric


def main(cl_arguments):
    """ Train a model for multitask-training."""
    cl_args = handle_arguments(cl_arguments)
    args = config.params_from_file(cl_args.config_file, cl_args.overrides)
    # Check for deprecated arg names
    check_arg_name(args)
    args, seed = initial_setup(args, cl_args)
    # Load tasks
    log.info("Loading tasks...")
    start_time = time.time()
    cuda_device = parse_cuda_list_arg(args.cuda)
    pretrain_tasks, target_tasks, vocab, word_embs = build_tasks(args, cuda_device)
    tasks = sorted(set(pretrain_tasks + target_tasks), key=lambda x: x.name)
    log.info("\tFinished loading tasks in %.3fs", time.time() - start_time)
    log.info("\t Tasks: {}".format([task.name for task in tasks]))
    # Build model
    log.info("Building model...")
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks, cuda_device)
    log.info("Finished building model in %.3fs", time.time() - start_time)

    # Start Tensorboard if requested
    if cl_args.tensorboard:
        tb_logdir = os.path.join(args.run_dir, "tensorboard")
        _run_background_tensorboard(tb_logdir, cl_args.tensorboard_port)

    check_configurations(args, pretrain_tasks, target_tasks)
    if args.do_pretrain:
        # Train on pretrain tasks
        log.info("Training...")
        stop_metric = get_pretrain_stop_metric(args.early_stopping_method, pretrain_tasks)
        should_decrease = (
            pretrain_tasks[0].val_metric_decreases if len(pretrain_tasks) == 1 else False
        )
        trainer, _, opt_params, schd_params = build_trainer(
            args, cuda_device, [], model, args.run_dir, sho
        uld_decrease, phase="pretrain"
        )
        to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        _ = trainer.train(
            pretrain_tasks,
            stop_metric,
            args.batch_size,
            args.weighting_method,
            args.scaling_method,
            to_train,
            opt_params,
            schd_params,
            args.load_model,
            phase="pretrain",
        )

    # For checkpointing logic
    if not args.do_target_task_training:
        strict = True
    else:
        strict = False

    if args.do_target_task_training:
        # Train on target tasks
        pre_target_train_path = setup_target_task_training(args, target_tasks, model, strict)
        target_tasks_to_train = copy.deepcopy(target_tasks)
        # Check for previous target train checkpoints
        task_to_restore, _, _ = check_for_previous_checkpoints(
            args.run_dir, target_tasks_to_train, "target_train", args.load_model
        )
        if task_to_restore is not None:
            # If there is a task to restore from, target train only on target tasks
            # including and following that task.
            last_task_index = [task.name for task in target_tasks_to_train].index(task_to_restore)
            target_tasks_to_train = target_tasks_to_train[last_task_index:]
        for task in target_tasks_to_train:
            # Skip tasks that should not be trained on.
            if task.eval_only_task:
                continue

            params_to_train = load_model_for_target_train_run(
                args, pre_target_train_path, model, strict, task, cuda_device
            )
            trainer, _, opt_params, schd_params = build_trainer(
                args,
                cuda_device,
                [task.name],
                model,
                args.run_dir,
                task.val_metric_decreases,
                phase="target_train",
            )

            _ = trainer.train(
                tasks=[task],
                stop_metric=task.val_metric,
                batch_size=args.batch_size,
                weighting_method=args.weighting_method,
                scaling_method=args.scaling_method,
                train_params=params_to_train,
                optimizer_params=opt_params,
                scheduler_params=schd_params,
                load_model=(task.name == task_to_restore),
                phase="target_train",
            )

    if args.do_full_eval:
        log.info("Evaluating...")
        splits_to_write = evaluate.parse_write_preds_arg(args.write_preds)

        # Evaluate on target_tasks.
        for task in target_tasks:
            # Find the task-specific best checkpoint to evaluate on.
            task_params = get_model_attribute(model, "_get_task_params", cuda_device)
            task_to_use = task_params(task.name).get("use_classifier", task.name)
            ckpt_path = get_best_checkpoint_path(args, "eval", task_to_use)
            assert ckpt_path is not None
            load_model_state(model, ckpt_path, cuda_device, skip_task_models=[], strict=strict)
            evaluate_and_write(args, model, [task], splits_to_write, cuda_device)

    if args.delete_checkpoints_when_done and not args.keep_all_checkpoints:
        log.info("Deleting all checkpoints.")
        delete_all_checkpoints(args.run_dir)

    log.info("Done!")


if __name__ == "__main__":
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
