import os

import torch

from transformers import AutoConfig

import jiant.proj.main.write_task_configs as write_task_configs
import jiant.proj.main.export_model as export_model
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as runscript
import jiant.shared.distributed as distributed
import jiant.utils.zconf as zconf
import jiant.utils.python.io as py_io
from jiant.utils.python.logic import replace_none
from jiant.utils.python.io import read_json


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    run_name = zconf.attr(type=str, required=True)
    exp_dir = zconf.attr(type=str, required=True)
    data_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    hf_pretrained_model_name_or_path = zconf.attr(type=str, required=True)
    model_weights_path = zconf.attr(type=str, default=None)
    model_cache_path = zconf.attr(type=str, default=None)

    # === Task parameters === #
    tasks = zconf.attr(type=str, default=None)
    train_tasks = zconf.attr(type=str, default=None)
    val_tasks = zconf.attr(type=str, default=None)
    test_tasks = zconf.attr(type=str, default=None)

    # === Misc parameters === #
    train_batch_size = zconf.attr(type=int, default=32)
    max_seq_length = zconf.attr(type=int, default=256)
    num_train_epochs = zconf.attr(type=float, default=3)
    train_examples_cap = zconf.attr(type=int, default=None)
    create_config = zconf.attr(action="store_true")

    # === Running Setup === #
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

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    optimizer_type = zconf.attr(default="adam", type=str)

    # === Specialized config === #
    no_cuda = zconf.attr(action="store_true")
    fp16 = zconf.attr(action="store_true")
    fp16_opt_level = zconf.attr(default="O1", type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default="", type=str)
    server_port = zconf.attr(default="", type=str)

    def _post_init(self):
        assert self.tasks or (
            self.train_tasks or self.val_tasks or self.test_tasks
        ), "Must include tasks or one of train_tasks, val_tasks, tests_tasks"
        if self.tasks and (self.train_tasks or self.val_tasks or self.test_tasks):
            assert (
                ([self.tasks] == self.train_tasks)
                and ([self.tasks] == self.val_tasks)
                and ([self.tasks] == self.test_tasks)
            ), "Tasks must be same as train_tasks/val_tasks/test_tasks if both are present"
        if self.tasks:
            self.train_tasks = self.tasks
            self.val_tasks = self.tasks
            self.test_tasks = self.tasks
        self.train_tasks = self.train_tasks.split(",")
        self.val_tasks = self.val_tasks.split(",")
        self.test_tasks = self.test_tasks.split(",")


def create_and_write_task_configs(task_name_list, data_dir, task_config_base_path):
    os.makedirs(task_config_base_path, exist_ok=True)
    task_config_path_dict = {}
    for task_name in task_name_list:
        task_config_path = os.path.join(task_config_base_path, f"{task_name}_config.json")
        write_task_configs.create_and_write_task_config(
            task_name=task_name,
            task_data_dir=os.path.join(data_dir, task_name),
            task_config_path=task_config_path,
        )
        task_config_path_dict[task_name] = task_config_path
    return task_config_path_dict


def run_simple(args: RunConfiguration, with_continue: bool = False):
    hf_config = AutoConfig.from_pretrained(args.hf_pretrained_model_name_or_path)
    model_cache_path = replace_none(
        args.model_cache_path, default=os.path.join(args.exp_dir, "models")
    )

    with distributed.only_first_process(local_rank=args.local_rank):
        # === Step 1: Write task configs based on templates === #
        full_task_name_list = sorted(list(set(args.train_tasks + args.val_tasks + args.test_tasks)))
        task_config_path_dict = {}
        if args.create_config:
            task_config_path_dict = create_and_write_task_configs(
                task_name_list=full_task_name_list,
                data_dir=args.data_dir,
                task_config_base_path=os.path.join(args.data_dir, "configs"),
            )
        else:
            for task_name in full_task_name_list:
                task_config_path_dict[task_name] = os.path.join(
                    args.data_dir, "configs", f"{task_name}_config.json"
                )

        # === Step 2: Download models === #
        if not os.path.exists(os.path.join(model_cache_path, hf_config.model_type)):
            print("Downloading model")
            export_model.export_model(
                hf_pretrained_model_name_or_path=args.hf_pretrained_model_name_or_path,
                output_base_path=os.path.join(model_cache_path, hf_config.model_type),
            )

        # === Step 3: Tokenize and cache === #
        phase_task_dict = {
            "train": args.train_tasks,
            "val": args.val_tasks,
            "test": args.test_tasks,
        }
        for task_name in full_task_name_list:
            phases_to_do = []
            for phase, phase_task_list in phase_task_dict.items():
                if task_name in phase_task_list and not os.path.exists(
                    os.path.join(args.exp_dir, "cache", hf_config.model_type, task_name, phase)
                ):
                    config = read_json(task_config_path_dict[task_name])
                    if phase in config["paths"]:
                        phases_to_do.append(phase)
                    else:
                        phase_task_list.remove(task_name)
            if not phases_to_do:
                continue
            print(f"Tokenizing Task '{task_name}' for phases '{','.join(phases_to_do)}'")
            tokenize_and_cache.main(
                tokenize_and_cache.RunConfiguration(
                    task_config_path=task_config_path_dict[task_name],
                    hf_pretrained_model_name_or_path=args.hf_pretrained_model_name_or_path,
                    output_dir=os.path.join(args.exp_dir, "cache", hf_config.model_type, task_name),
                    phases=phases_to_do,
                    # TODO: Need a strategy for task-specific max_seq_length issues (issue #1176)
                    max_seq_length=args.max_seq_length,
                    smart_truncate=True,
                    do_iter=True,
                )
            )

    # === Step 4: Generate jiant_task_container_config === #
    # We'll do this with a configurator. Creating a jiant_task_config has a surprising
    # number of moving parts.
    jiant_task_container_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path=os.path.join(args.data_dir, "configs"),
        task_cache_base_path=os.path.join(args.exp_dir, "cache", hf_config.model_type),
        train_task_name_list=args.train_tasks,
        val_task_name_list=args.val_tasks,
        test_task_name_list=args.test_tasks,
        train_batch_size=args.train_batch_size,
        eval_batch_multiplier=2,
        epochs=args.num_train_epochs,
        num_gpus=torch.cuda.device_count(),
        train_examples_cap=args.train_examples_cap,
    ).create_config()
    os.makedirs(os.path.join(args.exp_dir, "run_configs"), exist_ok=True)
    jiant_task_container_config_path = os.path.join(
        args.exp_dir, "run_configs", f"{args.run_name}_config.json"
    )
    py_io.write_json(jiant_task_container_config, path=jiant_task_container_config_path)

    # === Step 5: Train/Eval! === #
    if args.model_weights_path:
        model_load_mode = "partial"
        model_weights_path = args.model_weights_path
    else:
        # From Transformers
        if any(task_name.startswith("mlm_") for task_name in full_task_name_list):
            model_load_mode = "from_transformers_with_mlm"
        else:
            model_load_mode = "from_transformers"
        model_weights_path = os.path.join(
            model_cache_path, hf_config.model_type, "model", "model.p"
        )
    run_output_dir = os.path.join(args.exp_dir, "runs", args.run_name)

    if (
        args.save_checkpoint_every_steps
        and os.path.exists(os.path.join(run_output_dir, "checkpoint.p"))
        and with_continue
    ):
        print("Resuming")
        checkpoint = torch.load(os.path.join(run_output_dir, "checkpoint.p"))
        run_args = runscript.RunConfiguration.from_dict(checkpoint["metadata"]["args"])
    else:
        print("Running from start")
        run_args = runscript.RunConfiguration(
            # === Required parameters === #
            jiant_task_container_config_path=jiant_task_container_config_path,
            output_dir=run_output_dir,
            # === Model parameters === #
            hf_pretrained_model_name_or_path=args.hf_pretrained_model_name_or_path,
            model_path=model_weights_path,
            model_config_path=os.path.join(
                model_cache_path, hf_config.model_type, "model", "config.json",
            ),
            model_load_mode=model_load_mode,
            # === Running Setup === #
            do_train=bool(args.train_tasks),
            do_val=bool(args.val_tasks),
            do_save=args.do_save,
            do_save_best=args.do_save_best,
            do_save_last=args.do_save_last,
            write_val_preds=args.write_val_preds,
            write_test_preds=args.write_test_preds,
            eval_every_steps=args.eval_every_steps,
            save_every_steps=args.save_every_steps,
            save_checkpoint_every_steps=args.save_checkpoint_every_steps,
            no_improvements_for_n_evals=args.no_improvements_for_n_evals,
            keep_checkpoint_when_done=args.keep_checkpoint_when_done,
            force_overwrite=args.force_overwrite,
            seed=args.seed,
            # === Training Learning Parameters === #
            learning_rate=args.learning_rate,
            adam_epsilon=args.adam_epsilon,
            max_grad_norm=args.max_grad_norm,
            optimizer_type=args.optimizer_type,
            # === Specialized config === #
            no_cuda=args.no_cuda,
            fp16=args.fp16,
            fp16_opt_level=args.fp16_opt_level,
            local_rank=args.local_rank,
            server_ip=args.server_ip,
            server_port=args.server_port,
        )
        checkpoint = None

    runscript.run_loop(args=run_args, checkpoint=checkpoint)
    py_io.write_file(args.to_json(), os.path.join(run_output_dir, "simple_run_config.json"))


def main():
    mode, cl_args = zconf.get_mode_and_cl_args()
    args = RunConfiguration.default_run_cli(cl_args=cl_args)
    if mode == "run":
        run_simple(args, with_continue=False)
    elif mode == "run_with_continue":
        run_simple(args, with_continue=True)
    else:
        raise zconf.ModeLookupError(mode)


if __name__ == "__main__":
    main()
