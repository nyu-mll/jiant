import os

import torch

import jiant.utils.zconf as zconf
import jiant.utils.python.io as py_io


class Registry:
    func_dict = {}

    @classmethod
    def register(cls, f):
        cls.func_dict[f.__name__] = f
        return f


def write_configs(config_dict, base_path, check_paths=True):
    os.makedirs(base_path, exist_ok=True)
    config_keys = [
        "task_config_path_dict",
        "task_cache_config_dict",
        "sampler_config",
        "global_train_config",
        "task_specific_configs_dict",
        "metric_aggregator_config",
        "taskmodels_config",
        "task_run_config",
    ]
    for path in config_dict["task_config_path_dict"].values():
        if check_paths:
            assert os.path.exists(path)
    for path_dict in config_dict["task_cache_config_dict"].values():
        for path in path_dict.values():
            if check_paths:
                assert os.path.exists(path)
    for config_key in config_keys:
        py_io.write_json(
            config_dict[config_key], os.path.join(base_path, f"{config_key}.json"),
        )
    py_io.write_json(config_dict, os.path.join(base_path, "full.json"))
    py_io.write_json(
        {
            f"{config_key}_path": os.path.join(base_path, f"{config_key}.json")
            for config_key in config_keys
        },
        path=os.path.join(base_path, "zz_full.json"),
    )


def write_configs_from_full(full_config_path):
    write_configs(
        config_dict=py_io.read_json(full_config_path), base_path=os.path.split(full_config_path)[0],
    )


def get_num_examples_from_cache(cache_path):
    cache_metadata_path = os.path.join(cache_path, "data_args.p")
    return torch.load(cache_metadata_path)["length"]


def cap_examples(num_examples, cap):
    if cap is None:
        return num_examples
    else:
        return min(num_examples, cap)


@Registry.register
def single_task_config(
    task_config_path,
    train_batch_size=None,
    task_cache_base_path=None,
    epochs=None,
    max_steps=None,
    task_cache_train_path=None,
    task_cache_val_path=None,
    task_cache_val_labels_path=None,
    eval_batch_multiplier=2,
    eval_batch_size=None,
    gradient_accumulation_steps=1,
    eval_subset_num=500,
    num_gpus=1,
    warmup_steps_proportion=0.1,
    phases=("train", "val"),
):
    task_config = py_io.read_json(os.path.expandvars(task_config_path))
    task_name = task_config["name"]

    do_train = "train" in phases
    do_val = "val" in phases

    cache_path_dict = {}
    if do_train:
        if task_cache_train_path is None:
            task_cache_train_path = os.path.join(task_cache_base_path, "train")
        cache_path_dict["train"] = os.path.expandvars(task_cache_train_path)

    if do_val:
        if task_cache_val_path is None:
            task_cache_val_path = os.path.join(task_cache_base_path, "val")
        if task_cache_val_labels_path is None:
            task_cache_val_labels_path = os.path.join(task_cache_base_path, "val_labels")
        cache_path_dict["val"] = os.path.expandvars(task_cache_val_path)
        cache_path_dict["val_labels"] = os.path.expandvars(task_cache_val_labels_path)

    if do_train:
        assert (epochs is None) != (max_steps is None)
        assert train_batch_size is not None
        effective_batch_size = train_batch_size * gradient_accumulation_steps * num_gpus
        num_training_examples = get_num_examples_from_cache(
            cache_path=os.path.expandvars(task_cache_train_path),
        )
        max_steps = num_training_examples * epochs // effective_batch_size
    else:
        max_steps = 0
        train_batch_size = 0

    if do_val:
        if eval_batch_size is None:
            assert train_batch_size is not None
            eval_batch_size = train_batch_size * eval_batch_multiplier

    config_dict = {
        "task_config_path_dict": {task_name: os.path.expandvars(task_config_path)},
        "task_cache_config_dict": {task_name: cache_path_dict},
        "sampler_config": {"sampler_type": "UniformMultiTaskSampler"},
        "global_train_config": {
            "max_steps": max_steps,
            "warmup_steps": int(max_steps * warmup_steps_proportion),
        },
        "task_specific_configs_dict": {
            task_name: {
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "eval_subset_num": eval_subset_num,
            },
        },
        "taskmodels_config": {"task_to_taskmodel_map": {task_name: task_name}},
        "task_run_config": {
            "train_task_list": [task_name] if do_train else [],
            "train_val_task_list": [task_name] if do_train else [],
            "val_task_list": [task_name] if do_val else [],
            "test_task_list": [],
        },
        "metric_aggregator_config": {"metric_aggregator_type": "EqualMetricAggregator"},
    }
    return config_dict


@Registry.register
def simple_multi_task_config(
    task_meta_config_dict,
    task_cache_dict,
    task_name_list=None,
    epochs=None,
    max_steps=None,
    num_gpus=1,
    train_examples_cap=None,
    warmup_steps_proportion=0.1,
):
    if isinstance(task_meta_config_dict, str):
        task_meta_config_dict = py_io.read_json(os.path.expandvars(task_meta_config_dict))
    if isinstance(task_cache_dict, str):
        task_cache_dict = py_io.read_json(os.path.expandvars(task_cache_dict))
    if task_name_list is None:
        task_name_list = sorted(list(task_meta_config_dict))

    assert (epochs is None) != (max_steps is None)

    # Proportional
    num_examples_dict = {}
    capped_num_examples_dict = {}
    max_steps_not_given = max_steps is None
    print(max_steps_not_given)
    if max_steps_not_given:
        assert isinstance(epochs, (int, float))
        max_steps = 0
    for task_name in task_name_list:
        effective_batch_size = (
            task_meta_config_dict[task_name]["train_batch_size"]
            * task_meta_config_dict[task_name]["gradient_accumulation_steps"]
            * num_gpus
        )
        num_examples = get_num_examples_from_cache(
            cache_path=os.path.expandvars(task_cache_dict[task_name]["train"]),
        )
        capped_num_examples = cap_examples(num_examples=num_examples, cap=train_examples_cap)
        num_examples_dict[task_name] = num_examples
        capped_num_examples_dict[task_name] = capped_num_examples
        if max_steps_not_given:
            max_steps += num_examples * epochs // effective_batch_size

    if train_examples_cap is None:
        sampler_config = {
            "sampler_type": "ProportionalMultiTaskSampler",
        }
    else:
        sampler_config = {
            "sampler_type": "SpecifiedProbMultiTaskSampler",
            "task_to_unweighted_probs": capped_num_examples_dict,
        }

    config_dict = {
        "task_config_path_dict": {
            task_name: os.path.expandvars(task_meta_config_dict[task_name]["config_path"])
            for task_name in task_name_list
        },
        "task_cache_config_dict": {
            task_name: {
                "train": os.path.expandvars(task_cache_dict[task_name]["train"]),
                "val": os.path.expandvars(task_cache_dict[task_name]["val"]),
                "val_labels": os.path.expandvars(task_cache_dict[task_name]["val_labels"]),
            }
            for task_name in task_name_list
        },
        "sampler_config": sampler_config,
        "global_train_config": {
            "max_steps": max_steps,
            "warmup_steps": int(max_steps * warmup_steps_proportion),
        },
        "task_specific_configs_dict": {
            task_name: {
                "train_batch_size": task_meta_config_dict[task_name]["train_batch_size"],
                "eval_batch_size": task_meta_config_dict[task_name]["eval_batch_size"],
                "gradient_accumulation_steps": task_meta_config_dict[task_name][
                    "gradient_accumulation_steps"
                ],
                "eval_subset_num": task_meta_config_dict[task_name]["eval_subset_num"],
            }
            for task_name in task_name_list
        },
        "taskmodels_config": {
            "task_to_taskmodel_map": {
                task_name: task_meta_config_dict[task_name]["task_to_taskmodel_map"]
                for task_name in task_name_list
            },
            "taskmodel_config_map": {task_name: None for task_name in task_name_list},
        },
        "task_run_config": {
            "train_task_list": task_name_list,
            "train_val_task_list": task_name_list,
            "val_task_list": task_name_list,
            "test_task_list": task_name_list,
        },
        "metric_aggregator_config": {"metric_aggregator_type": "EqualMetricAggregator"},
    }
    return config_dict


@zconf.run_config
class JsonRunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    func = zconf.attr(type=str, required=True)
    path = zconf.attr(type=str, required=True)
    output_base_path = zconf.attr(type=str, required=True)


def main():
    mode, cl_args = zconf.get_mode_and_cl_args()
    if mode == "json":
        args = JsonRunConfiguration.default_run_cli(cl_args=cl_args)
        config_dict = Registry.func_dict[args.func](**py_io.read_json(args.path))
        write_configs(
            config_dict=config_dict, base_path=args.output_base_path,
        )
    else:
        raise zconf.ModeLookupError(mode)


if __name__ == "__main__":
    main()
