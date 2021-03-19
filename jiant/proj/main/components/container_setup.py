import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import jiant.proj.main.components.task_sampler as jiant_task_sampler
import jiant.shared.caching as caching
import jiant.tasks as tasks
import jiant.utils.python.io as py_io
from jiant.utils.python.datastructures import ExtendedDataClassMixin


@dataclass
class TaskSpecificConfig(ExtendedDataClassMixin):
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    eval_subset_num: int


@dataclass
class GlobalTrainConfig(ExtendedDataClassMixin):
    max_steps: int
    warmup_steps: int


@dataclass
class TaskmodelsConfig(ExtendedDataClassMixin):
    task_to_taskmodel_map: Dict[str, str]
    taskmodel_config_map: Optional[Dict[str, Optional[Dict]]] = None

    def get_taskmodel_kwargs(self, taskmodel_name: str) -> Optional[Dict]:
        assert taskmodel_name in self.task_to_taskmodel_map.values()
        if self.taskmodel_config_map is None:
            return None
        elif taskmodel_name not in self.taskmodel_config_map:
            return None
        else:
            return self.taskmodel_config_map[taskmodel_name]


@dataclass
class TaskRunConfig(ExtendedDataClassMixin):
    train_task_list: List[str]
    train_val_task_list: List[str]
    val_task_list: List[str]
    test_task_list: List[str]


@dataclass
class JiantTaskContainer:
    task_dict: Dict[str, tasks.Task]
    task_sampler: jiant_task_sampler.BaseMultiTaskSampler
    task_cache_dict: Dict
    global_train_config: GlobalTrainConfig
    task_specific_configs: Dict[str, TaskSpecificConfig]
    taskmodels_config: TaskmodelsConfig
    task_run_config: TaskRunConfig
    metrics_aggregator: jiant_task_sampler.BaseMetricAggregator


def create_task_dict(task_config_dict: dict, verbose: bool = True) -> Dict[str, tasks.Task]:
    """Make map of task name to task instances from map of task name to task config file paths.

    Args:
        task_config_dict (Dict): map from task name to task config filepath.
        verbose (bool): True to print task config info.

    Returns:
        Dict mapping from task name to task instance.

    """
    task_dict = {}
    for task_name, task_config_path in task_config_dict.items():
        task = tasks.create_task_from_config_path(config_path=task_config_path, verbose=False)
        if not task.name == task_name:
            warnings.warn(
                "task {} from {} has conflicting names: {}/{}. Using {}".format(
                    task_name, task_config_path, task_name, task.name, task_name,
                )
            )
            task.name = task_name
        task_dict[task_name] = task
    if verbose:
        print("Creating Tasks:")
        for task_name, task_config_path in task_config_dict.items():
            task_class = task_dict[task_name].__class__.__name__
            print(f"    {task_name} ({task_class}): {task_config_path}")
    return task_dict


def create_task_cache_dict(task_cache_config_dict: Dict) -> Dict:
    """Takes a map of task cache configs, and returns map of instantiated task data cache objects.

    Notes:
        This function assumes that data is divided and stored according to phase where phase takes
        a value of train, val, val_labels, or test.

    Args:
        task_cache_config_dict (Dict[str, Dict[str, str]]): maps of task names to cache file dirs.

    Returns:
        Dict[str, Dict[str, ChunkedFilesDataCache]] mappings from task name to task cache objects.

    """
    task_cache_dict = {}
    for task_name, task_cache_config in task_cache_config_dict.items():
        single_task_cache_dict = {}
        for phase in ["train", "val", "val_labels", "test"]:
            if phase in task_cache_config:
                single_task_cache_dict[phase] = caching.ChunkedFilesDataCache(
                    task_cache_config[phase],
                )
        task_cache_dict[task_name] = single_task_cache_dict
    return task_cache_dict


def get_num_train_examples(task_cache_dict: Dict, train_task_list=None) -> Dict[str, int]:
    """Count training examples for tasks.

    Args:
        task_cache_dict: nested maps from task name to phases, and from phase to task cache object.
        train_task_list (List): list of task names for which to count training examples.

    Notes:
        If get_num_train_examples() is called without providing train_task_list, training examples
        for all tasks in task_cache_dict are counted by get_num_train_examples().

    Returns:
        Dict[str, int] mapping task names to the count of training examples for that task.

    """
    num_train_examples_dict = {}
    if train_task_list is None:
        train_task_list = list(task_cache_dict)
    for task_name in train_task_list:
        single_task_cache_dict = task_cache_dict[task_name]
        if "train" in single_task_cache_dict:
            num_train_examples_dict[task_name] = len(single_task_cache_dict["train"])
        else:
            num_train_examples_dict[task_name] = 0
    return num_train_examples_dict


def create_task_specific_configs(task_specific_configs_dict) -> Dict[str, TaskSpecificConfig]:
    """Takes task-specific configs, returns them as TaskSpecificConfig(s).

    Args:
        task_specific_configs_dict: map task name to map of task-specific config name to value.

    Raises:
        TypeError if task-specific config is not either a dict or a TaskSpecificConfig object.

    Returns:
        Dict[str, TaskSpecificConfig] map of task name to task-specific configs.

    """
    task_specific_configs = {}
    for k, v in task_specific_configs_dict.items():
        if isinstance(v, dict):
            v = TaskSpecificConfig.from_dict(v)
        elif isinstance(v, TaskSpecificConfig):
            pass
        else:
            raise TypeError(type(v))
        task_specific_configs[k] = v
    return task_specific_configs


def create_jiant_task_container(
    task_config_path_dict: Dict,
    task_cache_config_dict: Dict,
    sampler_config: Dict,
    global_train_config: Dict,
    task_specific_configs_dict: Dict,
    metric_aggregator_config: Dict,
    taskmodels_config: Dict,
    task_run_config: Dict,
    verbose: bool = True,
) -> JiantTaskContainer:
    """Read and interpret config files, initialize configuration objects, return JiantTaskContainer.

    Args:
        task_config_path_dict (Dict[str, str]): map of task names to task config files.
        task_cache_config_dict (Dict[str, str]): map of task names to cache file dirs.
        sampler_config (Dict): map containing sample config options, e.g., uniform task sampling.
        global_train_config (Dict): map of training configs shared by all tasks (e.g., max_steps).
        task_specific_configs_dict (Dict): map of maps mapping task names to task-specific options.
        metric_aggregator_config (Dict): map containing task metric aggregation options.
        taskmodels_config: maps mapping from tasks to models, and specifying task-model configs.
        task_run_config: config determining which tasks are used in which phase (e.g., train).
        verbose: True to print task info.

    Returns:
        JiantTaskContainer carrying components configured and set up pre-runner.

    """
    task_dict = create_task_dict(task_config_dict=task_config_path_dict, verbose=verbose)
    task_cache_dict = create_task_cache_dict(task_cache_config_dict=task_cache_config_dict)
    global_train_config = GlobalTrainConfig.from_dict(global_train_config)
    task_specific_config = create_task_specific_configs(
        task_specific_configs_dict=task_specific_configs_dict,
    )
    taskmodels_config = TaskmodelsConfig.from_dict(taskmodels_config)
    task_run_config = TaskRunConfig.from_dict(task_run_config)

    num_train_examples_dict = get_num_train_examples(
        task_cache_dict=task_cache_dict, train_task_list=task_run_config.train_task_list,
    )
    task_sampler = jiant_task_sampler.create_task_sampler(
        sampler_config=sampler_config,
        # task sampler samples only from the training tasks
        task_dict={
            task_name: task_dict[task_name] for task_name in task_run_config.train_task_list
        },
        task_to_num_examples_dict=num_train_examples_dict,
    )
    metric_aggregator = jiant_task_sampler.create_metric_aggregator(
        metric_aggregator_config=metric_aggregator_config,
    )
    return JiantTaskContainer(
        task_dict=task_dict,
        task_sampler=task_sampler,
        global_train_config=global_train_config,
        task_cache_dict=task_cache_dict,
        task_specific_configs=task_specific_config,
        taskmodels_config=taskmodels_config,
        task_run_config=task_run_config,
        metrics_aggregator=metric_aggregator,
    )


def create_jiant_task_container_from_dict(
    jiant_task_container_config_dict: Dict, verbose: bool = True
) -> JiantTaskContainer:
    return create_jiant_task_container(
        task_config_path_dict=jiant_task_container_config_dict["task_config_path_dict"],
        task_cache_config_dict=jiant_task_container_config_dict["task_cache_config_dict"],
        sampler_config=jiant_task_container_config_dict["sampler_config"],
        global_train_config=jiant_task_container_config_dict["global_train_config"],
        task_specific_configs_dict=jiant_task_container_config_dict["task_specific_configs_dict"],
        taskmodels_config=jiant_task_container_config_dict["taskmodels_config"],
        task_run_config=jiant_task_container_config_dict["task_run_config"],
        metric_aggregator_config=jiant_task_container_config_dict["metric_aggregator_config"],
        verbose=verbose,
    )


def create_jiant_task_container_from_json(
    jiant_task_container_config_path: str, verbose: bool = True
) -> JiantTaskContainer:
    return create_jiant_task_container_from_dict(
        jiant_task_container_config_dict=py_io.read_json(jiant_task_container_config_path),
        verbose=verbose,
    )
