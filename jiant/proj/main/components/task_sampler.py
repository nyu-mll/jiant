import abc
import numpy as np

from typing import Union, Optional, Dict


class BaseMultiTaskSampler(metaclass=abc.ABCMeta):
    def __init__(self, task_dict: dict, rng: Union[int, np.random.RandomState, None]):
        self.task_dict = task_dict
        if isinstance(rng, int) or rng is None:
            rng = np.random.RandomState(rng)
        self.rng = rng

    def pop(self):
        raise NotImplementedError()

    def iter(self):
        yield self.pop()


class UniformMultiTaskSampler(BaseMultiTaskSampler):
    def pop(self):
        task_name = self.rng.choice(list(self.task_dict))
        return task_name, self.task_dict[task_name]


class ProportionalMultiTaskSampler(BaseMultiTaskSampler):
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_num_examples_dict: dict,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_num_examples_dict.keys()
        self.task_to_examples_dict = task_to_num_examples_dict
        self.task_names = list(task_to_num_examples_dict.keys())
        self.task_num_examples = np.array([task_to_num_examples_dict[k] for k in self.task_names])
        self.task_p = self.task_num_examples / self.task_num_examples.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


class SpecifiedProbMultiTaskSampler(BaseMultiTaskSampler):
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_unweighted_probs: dict,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_unweighted_probs.keys()
        self.task_to_unweighted_probs = task_to_unweighted_probs
        self.task_names = list(task_to_unweighted_probs.keys())
        self.unweighted_probs_arr = np.array([task_to_unweighted_probs[k] for k in self.task_names])
        self.task_p = self.unweighted_probs_arr / self.unweighted_probs_arr.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


class TemperatureMultiTaskSampler(BaseMultiTaskSampler):
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_num_examples_dict: dict,
        temperature: float,
        examples_cap: Optional[int],
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_num_examples_dict.keys()
        self.task_to_num_examples_dict = task_to_num_examples_dict
        self.temperature = temperature
        self.examples_cap = examples_cap
        self.task_names = list(task_to_num_examples_dict.keys())
        self.task_num_examples = np.array([task_to_num_examples_dict[k] for k in self.task_names])
        raw_n = self.task_num_examples.clip(max=examples_cap) ** (1 / self.temperature)
        self.task_p = raw_n / raw_n.sum()

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p)
        return task_name, self.task_dict[task_name]


def create_task_sampler(
    sampler_config: dict, task_dict: dict, task_to_num_examples_dict: dict, rng=None
) -> BaseMultiTaskSampler:
    sampler_type = sampler_config["sampler_type"]
    if sampler_type == "UniformMultiTaskSampler":
        assert len(sampler_config) == 1
        return UniformMultiTaskSampler(task_dict=task_dict, rng=rng)
    elif sampler_type == "ProportionalMultiTaskSampler":
        assert len(sampler_config) == 1
        return ProportionalMultiTaskSampler(
            task_dict=task_dict, rng=rng, task_to_num_examples_dict=task_to_num_examples_dict,
        )
    elif sampler_type == "SpecifiedProbMultiTaskSampler":
        assert len(sampler_config) == 2
        return SpecifiedProbMultiTaskSampler(
            task_dict=task_dict,
            rng=rng,
            task_to_unweighted_probs=sampler_config["task_to_unweighted_probs"],
        )
    elif sampler_type == "TemperatureMultiTaskSampler":
        assert len(sampler_config) == 3
        return TemperatureMultiTaskSampler(
            task_dict=task_dict,
            rng=rng,
            task_to_num_examples_dict=task_to_num_examples_dict,
            temperature=sampler_config["temperature"],
            examples_cap=sampler_config["examples_cap"],
        )
    else:
        raise KeyError(sampler_type)


class BaseMetricAggregator(metaclass=abc.ABCMeta):
    def aggregate(self, major_metrics_dict: Dict[str, float]):
        raise NotImplementedError()


class EqualMetricAggregator(BaseMetricAggregator):
    def aggregate(self, major_metrics_dict: Dict[str, float]):
        return np.mean([x for x in major_metrics_dict.values()])


class WeightedMetricAggregator(BaseMetricAggregator):
    def __init__(self, weights_dict: Dict[str, float]):
        self.weights_dict = weights_dict
        self.total_weights = sum([x for x in weights_dict.values()])

    def aggregate(self, major_metrics_dict: Dict[str, float]):
        return (
            np.sum(
                [x * self.weights_dict[task_name] for task_name, x in major_metrics_dict.items()]
            )
            / self.total_weights
        )


def create_metric_aggregator(metric_aggregator_config: Dict) -> BaseMetricAggregator:
    metric_aggregator_type = metric_aggregator_config["metric_aggregator_type"]
    if metric_aggregator_type == "EqualMetricAggregator":
        assert len(metric_aggregator_config) == 1
        return EqualMetricAggregator()
    elif metric_aggregator_type == "WeightedMetricAggregator":
        assert len(metric_aggregator_config) == 2
        return WeightedMetricAggregator(weights_dict=metric_aggregator_config["weights_dict"])
    else:
        raise KeyError(metric_aggregator_type)


def compute_aggregate_major_metrics_from_results_dict(metrics_aggregator, results_dict):
    major_metrics_dict = {
        task_name: results["metrics"].major for task_name, results in results_dict.items()
    }
    return metrics_aggregator.aggregate(major_metrics_dict=major_metrics_dict)


def get_metrics_dict_from_results_dict(results_dict):
    return {task_name: results["metrics"].to_dict() for task_name, results in results_dict.items()}
