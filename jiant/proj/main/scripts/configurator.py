import math
import os

import torch

import jiant.utils.zconf as zconf
import jiant.utils.python.io as py_io
import jiant.utils.python.datastructures as py_datastructures


class Registry:
    configurator_dict = {}

    @classmethod
    def register(cls, configurator_class):
        cls.configurator_dict[configurator_class.__name__] = configurator_class
        return configurator_class

    @classmethod
    def get_configurator(cls, configurator_name):
        if configurator_name not in cls.configurator_dict:
            raise KeyError(
                f"Configurator {configurator_name} not found in "
                f"available configurators: {list(cls.configurator_dict)}"
            )
        return cls.configurator_dict[configurator_name]


def get_num_examples_from_cache(cache_path):
    cache_metadata_path = os.path.join(cache_path, "data_args.p")
    return torch.load(cache_metadata_path)["length"]


def cap_examples(num_examples, cap):
    if cap is None:
        return num_examples
    else:
        return min(num_examples, cap)


@Registry.register
@zconf.run_config
class SingleTaskConfigurator(zconf.RunConfig):
    """Single-task Configurator

    Required:
        task_name
        train_batch_size

    (Task config) Need one of:
        task_config_path
        task_config_base_path

    (Task cache) Need one of:
        task_cache_path
        task_cache_base_path

    (Eval batch size) Need one of:
        eval_batch_multiplier
        eval_batch_size

    (Computing max steps) Need one of:
        epochs
        max_steps
        (Set to 0 if not training)

    (Task name list) Specify at least one of:
        do_train
        do_val
        do_test

    Optional:
        gradient_accumulation_steps
        eval_subset_num
        num_gpus
        train_examples_cap
        warmup_steps_proportion
    """

    task_name = zconf.attr(type=str, default=None)
    task_config_base_path = zconf.attr(type=str, default=None)
    task_config_path = zconf.attr(type=str, default=None)
    task_cache_base_path = zconf.attr(type=str, default=None)
    task_cache_path = zconf.attr(type=str, default=None)
    do_train = zconf.attr(action="store_true")
    do_val = zconf.attr(action="store_true")
    do_test = zconf.attr(action="store_true")
    train_batch_size = zconf.attr(type=int, required=True)
    eval_batch_multiplier = zconf.attr(type=int, default=None)
    eval_batch_size = zconf.attr(type=int, default=None)
    gradient_accumulation_steps = zconf.attr(type=int, default=1)
    eval_subset_num = zconf.attr(type=int, default=500)
    epochs = zconf.attr(type=int, default=None)
    max_steps = zconf.attr(type=int, default=None)
    num_gpus = zconf.attr(type=int, default=None)
    warmup_steps_proportion = zconf.attr(type=float, default=0.1)

    def create_config(self):
        # === Get task config === #
        if self.task_config_path:
            assert self.task_config_base_path is None
            task_config_path = self.task_config_path
        elif self.task_config_base_path is not None:
            assert self.task_config_path is None
            task_config_path = os.path.join(
                self.task_config_base_path, f"{self.task_name}_config.json",
            )
        else:
            raise RuntimeError("Require either `task_config_path` or `task_config_base_path`")

        # === Get cache === #
        if self.task_cache_path is not None:
            assert self.task_cache_base_path is None
            task_cache_path = self.task_cache_path
        elif self.task_cache_base_path is not None:
            assert self.task_cache_path is None
            task_cache_path = os.path.join(self.task_cache_base_path, self.task_name)
        else:
            raise RuntimeError("Need `task_cache_path` or `task_cache_base_path`")
        task_cache_config = {}
        if self.do_train:
            task_cache_config["train"] = os.path.join(task_cache_path, "train")
        if self.do_val:
            task_cache_config["val"] = os.path.join(task_cache_path, "val")
            task_cache_config["val_labels"] = os.path.join(task_cache_path, "val_labels")
        if self.do_test:
            task_cache_config["test"] = os.path.join(task_cache_path, "test")
        for v in task_cache_config.values():
            assert os.path.exists(v)

        # === Compute training steps === #
        if not self.do_train:
            assert self.epochs is None
            assert self.max_steps is None
            max_steps = 0
        elif self.max_steps is not None:
            max_steps = self.max_steps
        elif self.epochs is not None:
            assert self.max_steps is None
            if self.num_gpus:
                # We multiply by num_gpus because 1 step is done across (potentially) multiple GPUs
                effective_batch_size = (
                    self.train_batch_size * self.gradient_accumulation_steps * self.num_gpus
                )
            else:
                effective_batch_size = self.train_batch_size * self.gradient_accumulation_steps
            num_examples = get_num_examples_from_cache(
                cache_path=os.path.expandvars(task_cache_config["train"]),
            )
            max_steps = self.epochs * math.ceil(num_examples / effective_batch_size)
        else:
            raise RuntimeError("Require either `epochs` or `max_steps`")

        # === Compute eval_batch_size === #
        if self.eval_batch_size is not None:
            assert self.eval_batch_multiplier is None
            eval_batch_size = self.eval_batch_size
        elif self.eval_batch_multiplier is not None:
            assert self.eval_batch_size is None
            eval_batch_size = self.train_batch_size * self.eval_batch_multiplier
        else:
            raise RuntimeError("Require either `eval_batch_size` or `eval_batch_multiplier`")

        # === Build configuration === #
        # Finally, we build our big config dictionary. Congrats!
        config_dict = {
            "task_config_path_dict": {self.task_name: task_config_path},
            "task_cache_config_dict": {self.task_name: task_cache_config},
            "sampler_config": {"sampler_type": "UniformMultiTaskSampler"},
            "global_train_config": {
                "max_steps": int(max_steps),
                "warmup_steps": int(max_steps * self.warmup_steps_proportion),
            },
            "task_specific_configs_dict": {
                self.task_name: {
                    "train_batch_size": self.train_batch_size,
                    "eval_batch_size": eval_batch_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "eval_subset_num": self.eval_subset_num,
                }
            },
            "taskmodels_config": {
                "task_to_taskmodel_map": {self.task_name: self.task_name},
                "taskmodel_config_map": {self.task_name: None},
            },
            "task_run_config": {
                "train_task_list": [self.task_name] if self.do_train else [],
                "train_val_task_list": [self.task_name] if self.do_train else [],
                "val_task_list": [self.task_name] if self.do_val else [],
                "test_task_list": [self.task_name] if self.do_test else [],
            },
            "metric_aggregator_config": {"metric_aggregator_type": "EqualMetricAggregator"},
        }
        return config_dict


@Registry.register
@zconf.run_config
class SimpleAPIMultiTaskConfigurator(zconf.RunConfig):
    """Multi-task Configurator designed for SimpleAPI

    For simplicity, we assume that certain properties are constant across all tasks:
      batch sizes and eval_subset_num.
    Any more complex, and the user is better off writing the config entirely on their own.

    Required:
        train_batch_size

    (Task config) Need one of:
        task_config_base_path
        task_config_path_dict

    (Eval batch size) Need one of:
        eval_batch_multiplier
        eval_batch_size

    (Task cache) Need one of:
        task_cache_base_path
        task_cache_config_dict

    (Computing max steps) Need one of:
        epochs
        max_steps
        (Set to 0 if not training)

    (Task name list) Specify at least one of:
        train_task_name_list
        train_val_task_name_list
        val_task_name_list
        test_task_name_list

    Optional:
        gradient_accumulation_steps
        eval_subset_num
        num_gpus
        train_examples_cap
        warmup_steps_proportion
    """

    task_config_base_path = zconf.attr(type=str, default=None)
    task_config_path_dict = zconf.attr(type=str, default=None)
    task_cache_base_path = zconf.attr(type=str, default=None)
    task_cache_config_dict = zconf.attr(type=str, default=None)
    train_task_name_list = zconf.attr(type=str, default=None)
    train_val_task_name_list = zconf.attr(type=str, default=None)
    val_task_name_list = zconf.attr(type=str, default=None)
    test_task_name_list = zconf.attr(type=str, default=None)
    train_batch_size = zconf.attr(type=int, required=True)
    eval_batch_multiplier = zconf.attr(type=int, default=None)
    eval_batch_size = zconf.attr(type=int, default=None)
    gradient_accumulation_steps = zconf.attr(type=int, default=1)
    eval_subset_num = zconf.attr(type=int, default=500)
    epochs = zconf.attr(type=int, default=None)
    max_steps = zconf.attr(type=int, default=None)
    num_gpus = zconf.attr(type=int, default=None)
    train_examples_cap = zconf.attr(type=int, default=None)
    warmup_steps_proportion = zconf.attr(type=float, default=0.1)

    @classmethod
    def parse_task_name_list(cls, task_name_list_arg):
        if task_name_list_arg is None:
            return []
        elif isinstance(task_name_list_arg, str):
            return task_name_list_arg.split(",")
        elif isinstance(task_name_list_arg, list):
            return task_name_list_arg
        else:
            raise TypeError(type(task_name_list_arg))

    def create_config(self):
        # === Gather task names === #
        # Get the full list of tasks across all phases
        task_name_list_dict = {
            "train": self.parse_task_name_list(self.train_task_name_list),
            "val": self.parse_task_name_list(self.val_task_name_list),
            "test": self.parse_task_name_list(self.test_task_name_list),
        }
        if self.train_val_task_name_list is None:
            task_name_list_dict["train_val"] = task_name_list_dict["train"]
        else:
            task_name_list_dict["train_val"] = self.parse_task_name_list(
                self.train_val_task_name_list
            )
        full_task_name_list = py_datastructures.get_unique_list_in_order(
            task_name_list_dict.values()
        )

        # === Gather task configs === #
        # Build task_config_path_dict, either via
        #   1. task_config_base_path: where all caches are contained within a given folder
        #   2. task_config_dict: explicitly provided dictionary to cache paths, potentially in JSON
        # Use dictionary directly, or load from JSON
        if self.task_config_base_path is not None:
            assert self.task_config_path_dict is None
            task_config_path_dict = {
                task_name: os.path.join(self.task_config_base_path, f"{task_name}_config.json")
                for task_name in full_task_name_list
            }
        else:
            if isinstance(self.task_config_path_dict, str):
                task_config_path_dict = py_io.read_json(
                    os.path.expandvars(self.task_config_path_dict)
                )
            else:
                task_config_path_dict = self.task_config_path_dict

        # === Gather cache === #
        # Build task_cache_base_path, either via
        #   1. task_cache_base_path: where all caches are contained within a given folder
        #   2. task_cache_config_dict: explicitly provided dictionary to cache paths,
        #                              potentially in JSON
        if self.task_cache_base_path is not None:
            assert self.task_cache_config_dict is None
            task_cache_config_dict = {}
            for task_name in full_task_name_list:
                task_cache_config_dict[task_name] = {}
                if task_name in task_name_list_dict["train"]:
                    task_cache_config_dict[task_name]["train"] = os.path.join(
                        self.task_cache_base_path, task_name, "train",
                    )
                if (
                    task_name in task_name_list_dict["train_val"]
                    or task_name in task_name_list_dict["val"]
                ):
                    task_cache_config_dict[task_name]["val"] = os.path.join(
                        self.task_cache_base_path, task_name, "val",
                    )
                    task_cache_config_dict[task_name]["val_labels"] = os.path.join(
                        self.task_cache_base_path, task_name, "val_labels",
                    )
                if task_name in task_name_list_dict["test"]:
                    task_cache_config_dict[task_name]["test"] = os.path.join(
                        self.task_cache_base_path, task_name, "test",
                    )
        elif isinstance(self.task_cache_config_dict, str):
            assert self.task_cache_base_path is None
            task_cache_config_dict = py_io.read_json(self.task_cache_config_dict)
        elif isinstance(task_config_path_dict, dict):
            task_cache_config_dict = self.task_cache_config_dict
        else:
            raise RuntimeError("Need 'task_cache_base_path' or 'task_cache_dict'")

        # === Compute training steps === #
        # Computing the number of training steps across multiple tasks is slightly
        # trickier than expected (unless max_steps is explicitly provided)
        # We need to get the number of examples for each task, divide by the
        # effective batch size (batch size per gpu * grad accum steps * number of gpus)
        # AND consider a common use-case where we cap the number of examples from a given task
        assert (self.epochs is None) != (
            self.max_steps is None
        ), "Specify only 'epochs' or 'max_steps'"
        num_examples_dict = {}
        capped_num_examples_dict = {}
        max_steps_not_given = self.max_steps is None
        if max_steps_not_given:
            assert isinstance(self.epochs, (int, float))
            max_steps = 0
        else:
            max_steps = self.max_steps
        for task_name in task_name_list_dict["train"]:
            if self.num_gpus:
                # We multiply by num_gpus because 1 step is done across (potentially) multiple GPUs
                effective_batch_size = (
                    self.train_batch_size * self.gradient_accumulation_steps * self.num_gpus
                )
            else:
                effective_batch_size = self.train_batch_size * self.gradient_accumulation_steps
            num_examples = get_num_examples_from_cache(
                cache_path=os.path.expandvars(task_cache_config_dict[task_name]["train"]),
            )
            capped_num_examples = cap_examples(
                num_examples=num_examples, cap=self.train_examples_cap
            )
            num_examples_dict[task_name] = num_examples
            capped_num_examples_dict[task_name] = capped_num_examples
            if max_steps_not_given:
                max_steps += self.epochs * math.ceil(capped_num_examples / effective_batch_size)

        # === Compute eval_batch_size === #
        # Eval batch size is often a multiple of train batch size,
        #   so we provide 2 ways to specify it
        assert (self.eval_batch_size is None) != (
            self.eval_batch_multiplier is None
        ), "Specify only 'eval_batch_size' or 'eval_batch_multiplier'"
        if self.eval_batch_multiplier is not None:
            eval_batch_size = self.train_batch_size * self.eval_batch_multiplier
        else:
            eval_batch_size = self.eval_batch_size

        # === Configure Sampler === #
        # We sample proportionally by default, unless our training examples are capped per task
        if self.train_examples_cap is None:
            sampler_config = {
                "sampler_type": "ProportionalMultiTaskSampler",
            }
        else:
            sampler_config = {
                "sampler_type": "SpecifiedProbMultiTaskSampler",
                "task_to_unweighted_probs": capped_num_examples_dict,
            }

        # === Build configuration === #
        # Finally, we build our big config dictionary. Congrats!
        config_dict = {
            "task_config_path_dict": task_config_path_dict,
            "task_cache_config_dict": task_cache_config_dict,
            "sampler_config": sampler_config,
            "global_train_config": {
                "max_steps": int(max_steps),
                "warmup_steps": int(max_steps * self.warmup_steps_proportion),
            },
            "task_specific_configs_dict": {
                task_name: {
                    "train_batch_size": self.train_batch_size,
                    "eval_batch_size": eval_batch_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "eval_subset_num": self.eval_subset_num,
                }
                for task_name in full_task_name_list
            },
            "taskmodels_config": {
                "task_to_taskmodel_map": {
                    task_name: task_name for task_name in full_task_name_list
                },
                "taskmodel_config_map": {task_name: None for task_name in full_task_name_list},
            },
            "task_run_config": {
                "train_task_list": task_name_list_dict["train"],
                "train_val_task_list": task_name_list_dict["train_val"],
                "val_task_list": task_name_list_dict["val"],
                "test_task_list": task_name_list_dict["test"],
            },
            "metric_aggregator_config": {"metric_aggregator_type": "EqualMetricAggregator"},
        }
        return config_dict


def main():
    full_cl_args = zconf.core.get_sys_args()
    assert len(full_cl_args) >= 1, "Require two arguments to start: configurator and out_path"
    configurator_name, config_path, *cl_args = full_cl_args
    configurator = Registry.get_configurator(configurator_name=configurator_name)
    config_dict = configurator.default_run_cli(cl_args=cl_args).create_config()
    os.makedirs(os.path.split(config_path)[0], exist_ok=True)
    py_io.write_json(config_dict, path=config_path)


if __name__ == "__main__":
    main()
