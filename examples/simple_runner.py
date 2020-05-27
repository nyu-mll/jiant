import os

import jiant.proj.main.components.container_setup as container_setup
import jiant.shared.initialization as initialization
import jiant.utils.zconf as zconf
from jiant.proj.main.runscript import setup_runner
from jiant.utils.display import show_json


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    working_dir = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="from_ptt", type=str)

    # === Running Setup === #
    force_overwrite = zconf.attr(action="store_true")
    seed = zconf.attr(type=int, default=-1)

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


def create_sample_jiant_task_container(working_dir):
    jiant_task_container = container_setup.create_jiant_task_container_from_dict({
        "task_config_path_dict": {
            "mnli": os.path.join(working_dir, "data/glue/configs/mnli.json"),
            "qnli": os.path.join(working_dir, "data/glue/configs/qnli.json"),
            "rte": os.path.join(working_dir, "data/glue/configs/rte.json"),
        },
        "task_cache_config_dict": {
            "mnli": {
                "train": os.path.join(working_dir, "cache/mnli/train"),
                "val": os.path.join(working_dir, "cache/mnli/val"),
                "val_labels": os.path.join(working_dir, "cache/mnli/val_labels"),
            },
            "qnli": {
                "train": os.path.join(working_dir, "cache/qnli/train"),
                "val": os.path.join(working_dir, "cache/qnli/val"),
                "val_labels": os.path.join(working_dir, "cache/qnli/val_labels"),
            },
            "rte": {
                "train": os.path.join(working_dir, "cache/rte/train"),
                "val": os.path.join(working_dir, "cache/rte/val"),
                "val_labels": os.path.join(working_dir, "cache/rte/val_labels"),
            },
        },
        "sampler_config": {
            "sampler_type": "UniformMultiTaskSampler",
        },
        "global_train_config": {
            "max_steps": 1000,
            "warmup_steps": 100,
        },
        "task_specific_configs_dict": {
            "mnli": {
                "train_batch_size": 8,
                "eval_batch_size": 32,
                "gradient_accumulation_steps": 1,
                "eval_subset_num": 500,
            },
            "qnli": {
                "train_batch_size": 8,
                "eval_batch_size": 32,
                "gradient_accumulation_steps": 1,
                "eval_subset_num": 500,
            },
            "rte": {
                "train_batch_size": 4,
                "eval_batch_size": 32,
                "gradient_accumulation_steps": 1,
                "eval_subset_num": 500,
            },
        },
        "taskmodels_config": {
            "task_to_taskmodel_map": {
                "mnli": "nli",
                "qnli": "nli",
                "rte": "rte",
            },
            "taskmodel_config_map": {
                "nli": None,
                "rte": None,
            }
        },
        "task_run_config": {
            "train_task_list": ["mnli", "qnli", "rte"],
            "train_val_task_list": ["mnli", "qnli", "rte"],
            "val_task_list": ["rte", "mnli"],
            "test_task_list": [],
        },
        "metric_aggregator_config": {
            "metric_aggregator_type": "EqualMetricAggregator",
        },
    })
    return jiant_task_container


def main(args: RunConfiguration):
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    jiant_task_container = create_sample_jiant_task_container(
        working_dir=args.working_dir,
    )
    runner = setup_runner(
        args=args,
        jiant_task_container=jiant_task_container,
        quick_init_out=quick_init_out,
        verbose=True,
    )
    runner.run_train()
    val_metrics = runner.run_val(jiant_task_container.task_run_config.val_task_list)
    show_json({
        task_name: task_result_dict["metrics"].to_dict()
        for task_name, task_result_dict in val_metrics.items()
    })


if __name__ == "__main__":
    main(args=RunConfiguration.default_run_cli())
