import os
import torch


import jiant.proj.main.modeling.model_setup as jiant_model_setup
import jiant.proj.main.runner as jiant_runner
import jiant.proj.main.components.container_setup as container_setup
import jiant.proj.main.metarunner as jiant_metarunner
import jiant.proj.main.components.evaluate as jiant_evaluate
import jiant.shared.initialization as initialization
import jiant.shared.distributed as distributed
import jiant.shared.model_setup as model_setup
import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    output_dir = zconf.attr(type=str, required=True)

    # === Task container parameters === #
    task_config_base_path = zconf.attr(type=str, default=None)
    task_cache_base_path = zconf.attr(type=str, default=None)
    train_tasks = zconf.attr(type=str, default=None)
    train_val_tasks = zconf.attr(type=str, default=None)
    val_tasks = zconf.attr(type=str, default=None)
    test_tasks = zconf.attr(type=str, default=None)
    batch_size_set = zconf.attr(type=str, default="base")
    effective_batch_size = zconf.attr(type=int, default=16)
    eval_batch_multiplier = zconf.attr(type=int, default=2)
    epochs = zconf.attr(type=int, default=None)
    max_steps = zconf.attr(type=int, default=None)
    warmup_steps_proportion = zconf.attr(type=float, default=0.1)
    sampler_type = zconf.attr(type=str, default="proportional_sampler")
    prob_sampler_task_probs = zconf.attr(type=str, default="")
    temperature_sampler_temperature = zconf.attr(type=float, default=1.0)
    temperature_sampler_examples_cap = zconf.attr(type=float, default=1000000000000)
    time_func_sampler_task_probs = zconf.attr(type=str, default="")
    multidds_sampler_lr = zconf.attr(type=float, default=1e-4)
    multidds_sampler_update_steps = zconf.attr(type=int, default=1000)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="from_transformers", type=str)

    # === Running Setup === #
    do_train = zconf.attr(action="store_true")
    do_val = zconf.attr(action="store_true")
    do_save = zconf.attr(action="store_true")
    do_save_last = zconf.attr(action="store_true")
    do_save_best = zconf.attr(action="store_true")
    write_val_preds = zconf.attr(action="store_true")
    write_test_preds = zconf.attr(action="store_true")
    eval_every_steps = zconf.attr(type=int, default=0)
    save_every_steps = zconf.attr(type=int, default=0)
    save_checkpoint_every_steps = zconf.attr(type=int, default=0)
    no_improvements_for_n_evals = zconf.attr(type=int, default=0)
    keep_checkpoint_when_done = zconf.attr(action="store_true")
    force_overwrite = zconf.attr(action="store_true")
    seed = zconf.attr(type=int, default=-1)
    multiple_checkpoints = zconf.attr(action="store_true")

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    optimizer_type = zconf.attr(default="adam", type=str)

    # Specialized config
    no_cuda = zconf.attr(action="store_true")
    fp16 = zconf.attr(action="store_true")
    fp16_opt_level = zconf.attr(default="O1", type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default="", type=str)
    server_port = zconf.attr(default="", type=str)

    # New args in transfer methods
    architecture = zconf.attr(default="default", type=str)
    adapter_fusion_attention_fusion = zconf.attr(action="store_true")
    adapter_fusion_freeze_transformer = zconf.attr(action="store_true")
    adapter_fusion_freeze_adapters = zconf.attr(action="store_true")
    sluice_task_a = zconf.attr(default="", type=str)
    sluice_task_b = zconf.attr(default="", type=str)
    sluice_num_subspaces = zconf.attr(default=4, type=int)
    sluice_init_var = zconf.attr(default=0.02, type=float)
    sluice_lr_multiplier = zconf.attr(default=1.0, type=float)
    transnorm_replacement = zconf.attr(action="store_true")
    transnorm_update_rate = zconf.attr(default=0.1, type=float)
    transnorm_skip = zconf.attr(action="store_true")
    weight_regularization_type = zconf.attr(default="", type=str)
    weight_regularization_coef = zconf.attr(default=0.0, type=float)
    runner_type = zconf.attr(default="default", type=str)
    reptile_inner_steps = zconf.attr(default=5, type=int)
    reptile_num_sampled_tasks = zconf.attr(default=8, type=int)
    target_task = zconf.attr(default="", type=str)

    multidds_skip_learner = zconf.attr(action="store_true")
    multidds_samper_update_freq = zconf.attr(default=1000, type=int)
    multidds_force_skip_tasks = zconf.attr(default="", type=str)
    multidds_fixed_sampling_task_prob = zconf.attr(default="", type=str)
    multidds_queue_size = zconf.attr(type=int, default=500)
    multidds_temperature = zconf.attr(type=float, default=0.1)
    multidds_accumulate_target_grad = zconf.attr(action="store_true")

    grad_sim_metric = zconf.attr(default="fisher_cos", type=str)
    grad_sim_nonlinear = zconf.attr(default="", type=str)
    grad_sim_smoothing = zconf.attr(default=0, type=float)
    grad_sim_indep = zconf.attr(action="store_true")

    dds_target_optimization_choice = zconf.attr(default="", type=str)
    dds_square_rewards = zconf.attr(action="store_true")
    dds_aprx_eps = zconf.attr(default=1e-4, type=float)


@zconf.run_config
class ResumeConfiguration(zconf.RunConfig):
    checkpoint_path = zconf.attr(type=str)


def setup_runner(
    args: RunConfiguration,
    jiant_task_container: container_setup.JiantTaskContainer,
    quick_init_out,
    verbose: bool = True,
) -> jiant_runner.JiantRunner:
    """Setup jiant model, optimizer, and runner, and return runner.

    Args:
        args (RunConfiguration): configuration carrying command line args specifying run params.
        jiant_task_container (container_setup.JiantTaskContainer): task and sampler configs.
        quick_init_out (QuickInitContainer): device (GPU/CPU) and logging configuration.
        verbose: If True, enables printing configuration info (to standard out).

    Returns:
        jiant_runner.JiantRunner

    """
    # TODO document why the distributed.only_first_process() context manager is being used here.
    with distributed.only_first_process(local_rank=args.local_rank):
        # load the model
        jiant_model = jiant_model_setup.setup_jiant_model(
            model_type=args.model_type,
            model_config_path=args.model_config_path,
            tokenizer_path=args.model_tokenizer_path,
            task_dict=jiant_task_container.task_dict,
            taskmodels_config=jiant_task_container.taskmodels_config,
            args=args,
        )
        jiant_model_setup.delegate_load_from_path(
            jiant_model=jiant_model, weights_path=args.model_path, load_mode=args.model_load_mode
        )

        if hasattr(jiant_model, "dds_model"):
            jiant_model.dds_model.encoder.load_state_dict(jiant_model.encoder.state_dict())

        jiant_model.to(quick_init_out.device)

    optimizer_scheduler = model_setup.create_optimizer(
        args=args,
        model=jiant_model,
        learning_rate=args.learning_rate,
        t_total=jiant_task_container.global_train_config.max_steps,
        warmup_steps=jiant_task_container.global_train_config.warmup_steps,
        warmup_proportion=None,
        optimizer_type=args.optimizer_type,
        verbose=verbose,
    )
    jiant_model, optimizer = model_setup.raw_special_model_setup(
        model=jiant_model,
        optimizer=optimizer_scheduler.optimizer,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        n_gpu=quick_init_out.n_gpu,
        local_rank=args.local_rank,
    )
    optimizer_scheduler.optimizer = optimizer
    rparams = jiant_runner.RunnerParameters(
        local_rank=args.local_rank,
        n_gpu=quick_init_out.n_gpu,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
    )
    if args.runner_type == "default":
        runner = jiant_runner.JiantRunner(
            jiant_task_container=jiant_task_container,
            jiant_model=jiant_model,
            optimizer_scheduler=optimizer_scheduler,
            device=quick_init_out.device,
            rparams=rparams,
            log_writer=quick_init_out.log_writer,
        )
    elif args.runner_type == "reptile":
        runner = jiant_runner.ReptileRunner(
            jiant_task_container=jiant_task_container,
            jiant_model=jiant_model,
            optimizer_scheduler=optimizer_scheduler,
            device=quick_init_out.device,
            rparams=rparams,
            log_writer=quick_init_out.log_writer,
            inner_steps=args.reptile_inner_steps,
            num_sampled_tasks=args.reptile_num_sampled_tasks,
        )
    elif args.runner_type == "multidds":
        runner = jiant_runner.MultiDDSRunner(
            jiant_task_container=jiant_task_container,
            jiant_model=jiant_model,
            optimizer_scheduler=optimizer_scheduler,
            device=quick_init_out.device,
            rparams=rparams,
            log_writer=quick_init_out.log_writer,
            sampler_update_freq=args.multidds_samper_update_freq,
            target_task=args.target_task,
            accumulate_target_grad=args.multidds_accumulate_target_grad,
            output_dir=args.output_dir,
        )
    elif args.runner_type == "dds":
        runner = jiant_runner.DDSRunner(
            jiant_task_container=jiant_task_container,
            jiant_model=jiant_model,
            optimizer_scheduler=optimizer_scheduler,
            device=quick_init_out.device,
            rparams=rparams,
            log_writer=quick_init_out.log_writer,
            target_task=args.target_task,
            output_dir=args.output_dir,
            target_optimization_choice=args.dds_target_optimization_choice,
            square_rewards=args.dds_square_rewards,
            aprx_eps=args.dds_aprx_eps,
        )
    elif args.runner_type == "grad_sim":
        runner = jiant_runner.GradSimRunner(
            jiant_task_container=jiant_task_container,
            jiant_model=jiant_model,
            optimizer_scheduler=optimizer_scheduler,
            device=quick_init_out.device,
            rparams=rparams,
            log_writer=quick_init_out.log_writer,
            independent_param=args.grad_sim_indep,
            smoothing=args.grad_sim_smoothing,
            target_task=args.target_task,
        )
    elif args.runner_type == "distill":
        raise NotImplementedError
    elif args.runner_type == "l2tww":
        raise NotImplementedError
    return runner


def run_loop(args: RunConfiguration, checkpoint=None):
    is_resumed = checkpoint is not None
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    print(quick_init_out.n_gpu)
    with quick_init_out.log_writer.log_context():
        jiant_task_container = container_setup.create_jiant_task_container_from_args(args)
        runner = setup_runner(
            args=args,
            jiant_task_container=jiant_task_container,
            quick_init_out=quick_init_out,
            verbose=True,
        )
        if is_resumed:
            runner.load_state(checkpoint["runner_state"])
            del checkpoint["runner_state"]
        checkpoint_saver = jiant_runner.CheckpointSaver(
            metadata={"args": args.to_dict()},
            output_dir=args.output_dir,
            multiple_checkpoints=args.multiple_checkpoints,
        )
        if args.do_train:
            metarunner = jiant_metarunner.JiantMetarunner(
                runner=runner,
                save_every_steps=args.save_every_steps,
                eval_every_steps=args.eval_every_steps,
                save_checkpoint_every_steps=args.save_checkpoint_every_steps,
                no_improvements_for_n_evals=args.no_improvements_for_n_evals,
                checkpoint_saver=checkpoint_saver,
                output_dir=args.output_dir,
                verbose=True,
                save_best_model=args.do_save or args.do_save_best,
                save_last_model=args.do_save or args.do_save_last,
                load_best_model=True,
                log_writer=quick_init_out.log_writer,
            )
            if is_resumed:
                metarunner.load_state(checkpoint["metarunner_state"])
                del checkpoint["metarunner_state"]
            metarunner.run_train_loop()

        if args.do_val:
            val_results_dict = runner.run_val(
                task_name_list=runner.jiant_task_container.task_run_config.val_task_list,
                return_preds=args.write_val_preds,
            )
            jiant_evaluate.write_val_results(
                val_results_dict=val_results_dict,
                metrics_aggregator=runner.jiant_task_container.metrics_aggregator,
                output_dir=args.output_dir,
                verbose=True,
            )
            if args.write_val_preds:
                jiant_evaluate.write_preds(
                    eval_results_dict=val_results_dict,
                    path=os.path.join(args.output_dir, "val_preds.p"),
                )
        else:
            assert not args.write_val_preds

        if args.write_test_preds:
            test_results_dict = runner.run_test(
                task_name_list=runner.jiant_task_container.task_run_config.test_task_list,
            )
            jiant_evaluate.write_preds(
                eval_results_dict=test_results_dict,
                path=os.path.join(args.output_dir, "test_preds.p"),
            )

    if (
        not args.keep_checkpoint_when_done
        and args.save_checkpoint_every_steps
        and os.path.exists(os.path.join(args.output_dir, "checkpoint.p"))
    ):
        os.remove(os.path.join(args.output_dir, "checkpoint.p"))

    py_io.write_file("DONE", os.path.join(args.output_dir, "done_file"))


def run_resume(args: ResumeConfiguration):
    resume(checkpoint_path=args.checkpoint_path)


def resume(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    args = RunConfiguration.from_dict(checkpoint["metadata"]["args"])
    run_loop(args=args, checkpoint=checkpoint)


def run_with_continue(cl_args):
    run_args = RunConfiguration.default_run_cli(cl_args=cl_args)
    if not run_args.force_overwrite and (
        os.path.exists(os.path.join(run_args.output_dir, "done_file"))
        or os.path.exists(os.path.join(run_args.output_dir, "val_metrics.json"))
    ):
        print("Already Done")
        return
    elif run_args.save_checkpoint_every_steps and os.path.exists(
        os.path.join(run_args.output_dir, "checkpoint.p")
    ):
        print("Resuming")
        resume(os.path.join(run_args.output_dir, "checkpoint.p"))
    else:
        print("Running from start")
        run_loop(args=run_args)


def main():
    mode, cl_args = zconf.get_mode_and_cl_args()
    if mode == "run":
        run_loop(RunConfiguration.default_run_cli(cl_args=cl_args))
    elif mode == "continue":
        run_resume(ResumeConfiguration.default_run_cli(cl_args=cl_args))
    elif mode == "run_with_continue":
        run_with_continue(cl_args=cl_args)
    else:
        raise zconf.ModeLookupError(mode)


if __name__ == "__main__":
    main()
