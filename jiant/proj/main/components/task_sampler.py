import abc
import numexpr
import numpy as np
import torch

from typing import Union, Optional, Dict, List, Tuple


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


class MultiDDSSampler(BaseMultiTaskSampler):
    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        skip_learner: bool = False,
        queue_size: int = None,
        temperature: float = 0.1,
        sampler_lr: float = None,
        sampler_update_steps: int = None,
        sampler_force_skip_tasks: List = None,
        fixed_sampling_task_prob: Tuple[
            str, float
        ] = None,  # Todo: deprecate sampler_force_skip_tasks
    ):
        super().__init__(task_dict=task_dict, rng=rng)

        if skip_learner:
            assert queue_size is not None
        else:
            assert sampler_lr is not None
            assert sampler_update_steps is not None

        self.skip_learner = skip_learner
        self.sampler_force_skip_tasks = sampler_force_skip_tasks
        if self.sampler_force_skip_tasks:
            self.skip_tasks_mask = torch.BoolTensor(
                [task_name in sampler_force_skip_tasks for task_name in self.task_dict]
            )
        self.fixed_sampling_task_prob = fixed_sampling_task_prob
        if self.fixed_sampling_task_prob:
            self.fixed_sampling_mask = torch.BoolTensor(
                [task_name == fixed_sampling_task_prob[0] for task_name in self.task_dict]
            )
            self.fixed_sampling_prob = fixed_sampling_task_prob[1]

        if not self.skip_learner:
            with torch.no_grad():
                # start from uniform distribution
                initial_weight = torch.FloatTensor([1.0 for k in self.task_dict])
                if torch.cuda.is_available():
                    initial_weight = initial_weight.cuda()
                    if self.sampler_force_skip_tasks:
                        self.skip_tasks_mask = self.skip_tasks_mask.cuda()
                    if self.fixed_sampling_task_prob:
                        self.fixed_sampling_mask = self.fixed_sampling_mask.cuda()
                if self.sampler_force_skip_tasks:
                    initial_weight = initial_weight.masked_fill(self.skip_tasks_mask, -float("Inf"))
                self.sampler_weight = initial_weight.detach()
                self.sampler_weight.requires_grad = True
            self.sampler_optimizer = torch.optim.Adam([self.sampler_weight], sampler_lr)
            self.sampler_update_steps = sampler_update_steps
        else:
            self.list_of_queues = [[] for k in self.task_dict]
            self.queue_size = queue_size
            self.temperature = temperature

        self.task_names = list(task_dict.keys())

    def task_p(self):
        if not self.skip_learner:
            return torch.softmax(self.sampler_weight, dim=0)
        else:
            task_scores = torch.tensor(
                [sum(queue) / len(queue) if queue else 0.0 for queue in self.list_of_queues]
            )
            if self.sampler_force_skip_tasks:
                task_scores = task_scores.masked_fill(self.skip_tasks_mask, -float("Inf"))
            if self.fixed_sampling_task_prob is not None:
                task_scores = task_scores.masked_fill(self.fixed_sampling_mask, -float("Inf"))
            task_probs = torch.softmax(task_scores / self.temperature, dim=0)
            if self.fixed_sampling_task_prob is not None:
                task_probs = task_probs * (1 - self.fixed_sampling_prob)
                task_probs = task_probs.masked_fill(
                    self.fixed_sampling_mask, self.fixed_sampling_prob
                )
            return task_probs

    def pop(self):
        task_name = self.rng.choice(self.task_names, p=self.task_p().detach().cpu().numpy())
        return task_name, self.task_dict[task_name]

    def update_sampler(self, reward: torch.FloatTensor):
        if not self.skip_learner:
            for step in range(self.sampler_update_steps):
                rl_loss = -(self.task_p() * reward).mean()
                rl_loss.backward()
                self.sampler_optimizer.step()
                self.sampler_optimizer.zero_grad()
        else:
            task_wise_rewards = [float(e) for e in reward.detach().cpu().numpy()]
            for queue, reward in zip(self.list_of_queues, task_wise_rewards):
                queue.append(reward)
                if len(queue) > self.queue_size:
                    queue.pop(0)


class TimeDependentProbMultiTaskSampler(BaseMultiTaskSampler):
    """Multi-task Sampler with different task sampling probabilities over time

    We describe the individual unnormalized probabilities using numexpr expressions,
    using t as the variable, e.g.:
    * 1             (constant)
    * 2 * t         (linear)
    * 1/sqrt(t)     (inverse square-root)

    These are computed for all tasks for each time step, and then normalized to sum to 1.

    Attributes:
        task_dict: Dictionary of tasks
        rng: Random seed, or NumPy RandomState for sampling
        task_to_unnormalized_prob_funcs_dict: map from task names to strings, which are
                                              numexpr expressions
        max_steps: Maximum number of steps allows (in the case where some functions
                   are not valid after a given t.
    """

    def __init__(
        self,
        task_dict: dict,
        rng: Union[int, np.random.RandomState],
        task_to_unnormalized_prob_funcs_dict: dict,
        max_steps: Optional[int] = None,
    ):
        super().__init__(task_dict=task_dict, rng=rng)
        assert task_dict.keys() == task_to_unnormalized_prob_funcs_dict.keys()
        self.task_to_unnormalized_prob_funcs_dict = task_to_unnormalized_prob_funcs_dict
        self.max_steps = max_steps

        self.task_names = list(task_to_unnormalized_prob_funcs_dict.keys())
        self.steps = 0

    def pop(self):
        if self.max_steps is not None and self.steps >= self.max_steps:
            raise IndexError(f"steps ({self.steps}) > max_steps ({self.max_steps})")
        task_name = self.rng.choice(self.task_names, p=self.get_task_p(self.steps))
        self.steps += 1
        return task_name, self.task_dict[task_name]

    def get_task_p(self, steps=None) -> np.ndarray:
        p_ls = np.empty(len(self.task_names))

        # t is the variable in the numexpr expression
        t = steps if steps is not None else self.steps

        for i, task_name in enumerate(self.task_names):
            p_ls[i] = numexpr.evaluate(
                self.task_to_unnormalized_prob_funcs_dict[task_name], local_dict={"t": t},
            )
        p_ls /= p_ls.sum()
        return p_ls

    def reset_counter(self):
        self.steps = 0


def create_task_sampler(
    sampler_config: dict, task_dict: dict, task_to_num_examples_dict: dict, rng=None
) -> BaseMultiTaskSampler:
    """Perform basic config validation, then instantiate and return the specified multitask sampler.

    Args:
        sampler_config (Dict): map containing sample config options.
        task_dict (Dict[str, Task]): map from task name to task instance.
        task_to_num_examples_dict (Dict[str, int]): map task names to counts of training examples.
        rng (Union[int, np.random.RandomState, None]): random state to seed sampler.

    Raises:
        KeyError if invalid sampler type argument is provided in the sampler config.

    Returns:
        Subclass of BaseMultiTaskSampler.

    """
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
    elif sampler_type == "TimeDependentProbMultiTaskSampler":
        assert len(sampler_config) == 3
        return TimeDependentProbMultiTaskSampler(
            task_dict=task_dict,
            rng=rng,
            task_to_unnormalized_prob_funcs_dict=sampler_config[
                "task_to_unnormalized_prob_funcs_dict"
            ],
            max_steps=sampler_config["max_steps"],
        )
    elif sampler_type == "MultiDDSSampler":
        return MultiDDSSampler(
            task_dict=task_dict,
            rng=rng,
            skip_learner=sampler_config["skip_learner"],
            sampler_lr=sampler_config["sampler_lr"],
            temperature=sampler_config["temperature"],
            sampler_update_steps=sampler_config["sampler_update_steps"],
            sampler_force_skip_tasks=sampler_config["sampler_force_skip_tasks"],
            fixed_sampling_task_prob=sampler_config["fixed_sampling_task_prob"],
            queue_size=sampler_config["queue_size"],
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
    """Perform basic config validation, then instantiate and return the specified metric aggregator.

    Args:
        metric_aggregator_config (Dict): map containing metric aggregation options.

    Returns:
        Subclass of BaseMetricAggregator.

    """
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
