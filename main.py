"""Train a multi-task model using AllenNLP

To debug this, run with -m ipdb:

    python -m ipdb main.py --config_file ...
"""
# pylint: disable=no-member
import logging as log

log.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
)  # noqa

import argparse
import glob
import io
import os
import random
import subprocess
import sys
import time
import copy
import torch
import jsondiff

from src import evaluate
from src.models import build_model
from src.preprocess import build_tasks
from src import tasks as tasks_module
from src.tasks.tasks import GLUEDiagnosticTask
from src.trainer import build_trainer
from src.utils import config
from src.utils.utils import (
    assert_for_log,
    check_arg_name,
    load_model_state,
    maybe_make_dir,
    parse_json_diff,
    sort_param_recursive,
    select_relevant_print_args,
)
import jsondiff

# Global notification handler, can be accessed outside main() during exception handling.
EMAIL_NOTIFIER = None


def handle_arguments(cl_arguments):
    parser = argparse.ArgumentParser(description="")
    # Configuration files
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        nargs="+",
        default="config/defaults.conf",
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
    Saves model states from pretraining if applicable, and
    loads the correct model state for the target task training
    stage.

    Parameters
    ----------------
    args: Params object
    target_tasks: list of target Task objects
    mdoel: a MultiTaskModel object

    Returns
    ----------------
    task_names_to_avoid_loading: list of strings, if we don't allow for
    use of pretrained target specific module parameters, then this list will
    consist of all the task names so that we avoid loading the
    pretrained parameters. Else, it will be an empty list.
    """
    if args.do_target_task_training and not args.allow_reuse_of_pretraining_parameters:
        # If we're training models for evaluation, which is always done from scratch with a fresh
        # optimizer, we shouldn't load parameters for those models.
        # Usually, there won't be trained parameters to skip, but this can happen if a run is
        # killed during the do_target_task_training phase.
        task_names_to_avoid_loading = [task.name for task in target_tasks]
    else:
        task_names_to_avoid_loading = []

    if not args.load_target_train_checkpoint == "none":
        # This is to load a particular target train checkpoint.
        log.info("Loading existing model from %s...", args.load_target_train_checkpoint)
        load_model_state(
            model,
            args.load_target_train_checkpoint,
            args.cuda,
            task_names_to_avoid_loading,
            strict=strict,
        )
    else:
        # Look for target train checkpoints (available only if we're restoring from a run that
        # already finished), then look for training checkpoints.

        best_path = get_best_checkpoint_path(args.run_dir, "target_train")
        if best_path:
            load_model_state(
                model, best_path, args.cuda, task_names_to_avoid_loading, strict=strict
            )
        else:
            if args.do_pretrain == 1:
                best_pretrain = get_best_checkpoint_path(args.run_dir, "pretrain")
                if best_pretrain:
                    load_model_state(
                        model, best_pretrain, args.cuda, task_names_to_avoid_loading, strict=strict
                    )
            else:
                assert_for_log(
                    args.allow_untrained_encoder_parameters, "No best checkpoint found to evaluate."
                )

                if args.transfer_paradigm == "finetune":
                    # We want to do target training without pretraining, thus
                    # we need to first create a checkpoint to come back to for each of
                    # the target tasks to finetune.
                    model_state = model.state_dict()
                    model_path = os.path.join(
                        args.run_dir, "model_state_untrained_pre_target_train.th"
                    )
                    torch.save(model_state, model_path)

            log.warning("Evaluating untrained encoder parameters!")
    return task_names_to_avoid_loading


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

    if args.load_target_train_checkpoint != "none":
        assert_for_log(
            os.path.exists(args.load_target_train_checkpoint),
            "Error: Attempting to load model from non-existent path: [%s]"
            % args.load_target_train_checkpoint,
        )
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
            args.pretrain_tasks != "none",
            "Error: Must specify at least one pretraining task: [%s]" % args.pretrain_tasks,
        )
        steps_log.write("Training model on tasks: %s \n" % args.pretrain_tasks)

    if args.do_target_task_training:
        assert_for_log(
            args.target_tasks != "none",
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
        steps_log.write("Evaluating model on tasks: %s \n" % args.target_tasks)

    log.info("Will run the following steps for this experiment:\n%s", steps_log.getvalue())
    steps_log.close()


def _log_git_info():
    try:
        c = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=10, stdout=subprocess.PIPE
        )
        git_branch_name = c.stdout.decode().strip()
        log.info("Git branch: %s", git_branch_name)
        c = subprocess.run(["git", "rev-parse", "HEAD"], timeout=10, stdout=subprocess.PIPE)
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


def get_best_checkpoint_path(run_dir, phase):
    """ Look in run_dir for model checkpoint to load.
    Hierarchy is
        1) best checkpoint from the phase so far
        2) if we do only target training without pretraining, then load checkpoint before 
        target training
        3) nothing found (empty string) """
    checkpoint = glob.glob(os.path.join(run_dir, "model_state_%s_epoch_*.best_macro.th" % phase))
    if len(checkpoint) == 0 and phase == "target_train":
        checkpoint = glob.glob(os.path.join(run_dir, "model_state_untrained_pre_target_train.th"))
    if len(checkpoint) > 0:
        assert_for_log(len(checkpoint) == 1, "Too many best checkpoints. Something is wrong.")
        return checkpoint[0]
    return None


def evaluate_and_write(args, model, tasks, splits_to_write):
    """ Evaluate a model on dev and/or test, then write predictions """
    val_results, val_preds = evaluate.evaluate(model, tasks, args.batch_size, args.cuda, "val")
    if "val" in splits_to_write:
        evaluate.write_preds(
            tasks, val_preds, args.run_dir, "val", strict_glue_format=args.write_strict_glue_format
        )
    if "test" in splits_to_write:
        _, te_preds = evaluate.evaluate(model, tasks, args.batch_size, args.cuda, "test")
        evaluate.write_preds(
            tasks, te_preds, args.run_dir, "test", strict_glue_format=args.write_strict_glue_format
        )
    run_name = args.get("run_name", os.path.basename(args.run_dir))

    results_tsv = os.path.join(args.exp_dir, "results.tsv")
    log.info("Writing results for split 'val' to %s", results_tsv)
    evaluate.write_results(val_results, results_tsv, run_name=run_name)


def initial_setup(args, cl_args):
    """
    Sets up email hook, creating seed, and cuda settings.

    Parameters
    ----------------
    args: Params object
    cl_args: list of arguments
    
    Returns
    ----------------
    tasks: list of Task objects
    pretrain_tasks: list of pretraining tasks
    target_tasks: list of target tasks
    vocab: list of vocab
    word_embs: loaded word embeddings, may be None if args.word_embs = none
    model: a MultiTaskModel object
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
        from src.utils import gcp

        gcp.configure_remote_logging(args.remote_log_name)

    if cl_args.notify:
        from src.utils import emails

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
    if args.cuda >= 0:
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

    return args, seed


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
    pretrain_tasks, target_tasks, vocab, word_embs = build_tasks(args)
    tasks = sorted(set(pretrain_tasks + target_tasks), key=lambda x: x.name)
    log.info("\tFinished loading tasks in %.3fs", time.time() - start_time)
    log.info("\t Tasks: {}".format([task.name for task in tasks]))

    # Build model
    log.info("Building model...")
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    log.info("Finished building model in %.3fs", time.time() - start_time)

    # Start Tensorboard if requested
    if cl_args.tensorboard:
        tb_logdir = os.path.join(args.run_dir, "tensorboard")
        _run_background_tensorboard(tb_logdir, cl_args.tensorboard_port)

    check_configurations(args, pretrain_tasks, target_tasks)

    if args.do_pretrain:
        # Train on pretrain tasks
        log.info("Training...")
        stop_metric = pretrain_tasks[0].val_metric if len(pretrain_tasks) == 1 else "macro_avg"
        should_decrease = (
            pretrain_tasks[0].val_metric_decreases if len(pretrain_tasks) == 1 else False
        )
        trainer, _, opt_params, schd_params = build_trainer(
            args, [], model, args.run_dir, should_decrease, phase="pretrain"
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
            args.shared_optimizer,
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
        task_names_to_avoid_loading = setup_target_task_training(args, target_tasks, model, strict)
        if args.transfer_paradigm == "frozen":
            # might be empty if elmo = 0. scalar_mix_0 should always be
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
        for task in target_tasks:
            # Skip diagnostic tasks b/c they should not be trained on
            if isinstance(task, GLUEDiagnosticTask):
                continue

            if args.transfer_paradigm == "finetune":
                # Train both the task specific models as well as sentence
                # encoder.
                to_train = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            else:  # args.transfer_paradigm == "frozen":
                # Only train task-specific module
                pred_module = getattr(model, "%s_mdl" % task.name)
                to_train = [(n, p) for n, p in pred_module.named_parameters() if p.requires_grad]
                to_train += elmo_scalars

            trainer, _, opt_params, schd_params = build_trainer(
                args,
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
                train_params=to_train,
                optimizer_params=opt_params,
                scheduler_params=schd_params,
                shared_optimizer=args.shared_optimizer,
                load_model=False,
                phase="target_train",
            )

            # Now that we've trained the task specific module for the task,
            # since we are accumulating the best parameters for
            # each task specific model, we allow for loading of the trained
            # module. We avoid loading the task specific modules at first
            # in order to make sure that any module-specific training from pretraining
            # step does not affect the target task training step.
            # This only affects for transfer_paradigm = frozen.
            if task.name in task_names_to_avoid_loading:
                task_names_to_avoid_loading.remove(task.name)

            if args.transfer_paradigm == "finetune":
                # Reload the original best model from before target-task
                # training since we specifically finetune for each task.
                pre_target_train = get_best_checkpoint_path(args.run_dir, "pretrain")

                load_model_state(
                    model, pre_finetune_path, args.cuda, skip_task_models=[], strict=strict
                )
            else:  # args.transfer_paradigm == "frozen":
                # Load the current overall best model.
                layer_path = get_best_checkpoint_path(args.run_dir, "target_train")
                assert layer_path, "No best checkpoint found."
                load_model_state(
                    model,
                    layer_path,
                    args.cuda,
                    strict=strict,
                    skip_task_models=task_names_to_avoid_loading,
                )

    if args.do_full_eval:
        # Evaluate
        log.info("Evaluating...")
        splits_to_write = evaluate.parse_write_preds_arg(args.write_preds)
        if args.transfer_paradigm == "finetune":
            for task in target_tasks:
                task_to_use = model._get_task_params(task.name).get("use_classifier", task.name)
                if task.name != task_to_use:
                    task_model_to_load = task_to_use
                else:
                    task_model_to_load = task.name

                # Special checkpointing logic here since we train the sentence encoder
                # and have a best set of sent encoder model weights per task.
                finetune_path = os.path.join(
                    args.run_dir, "model_state_%s_best.th" % task_model_to_load
                )
                if os.path.exists(finetune_path):
                    ckpt_path = finetune_path
                else:
                    if args.do_target_task_training == 0:
                        phase = "pretrain"
                    else:
                        phase = "target_train"
                    ckpt_path = get_best_checkpoint_path(args.run_dir, phase)

                assert "best" in ckpt_path
                load_model_state(model, ckpt_path, args.cuda, skip_task_models=[], strict=strict)

                evaluate_and_write(args, model, [task], splits_to_write)

        elif args.transfer_paradigm == "frozen":
            # Don't do any special checkpointing logic here
            # since model already has all the trained task specific modules.
            evaluate_and_write(args, model, target_tasks, splits_to_write)

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
