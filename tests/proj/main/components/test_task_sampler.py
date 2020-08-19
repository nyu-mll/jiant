import numpy as np
import pytest

import jiant.proj.main.components.task_sampler as task_sampler


def test_time_dependent_prob_multitask_sampler_const_p():
    sampler = task_sampler.TimeDependentProbMultiTaskSampler(
        task_dict={"rte": None, "mnli": None, "squad_v1": None,},
        rng=0,
        task_to_unnormalized_prob_funcs_dict={"rte": "1", "mnli": "1", "squad_v1": "1",},
    )
    gold_p = np.ones(3) / 3
    assert np.array_equal(sampler.get_task_p(0), gold_p)
    assert np.array_equal(sampler.get_task_p(500), gold_p)
    assert np.array_equal(sampler.get_task_p(999), gold_p)


def test_time_dependent_prob_multitask_sampler_variable_p():
    sampler = task_sampler.TimeDependentProbMultiTaskSampler(
        task_dict={"rte": None, "mnli": None, "squad_v1": None,},
        rng=0,
        task_to_unnormalized_prob_funcs_dict={
            "rte": "1",
            "mnli": "1 - t/1000",
            "squad_v1": "exp(t/1000)",
        },
    )
    assert np.array_equal(sampler.get_task_p(0), np.ones(3) / 3)
    assert np.allclose(sampler.get_task_p(500), np.array([0.31758924, 0.15879462, 0.52361614]))
    assert np.allclose(
        sampler.get_task_p(999), np.array([2.69065663e-01, 2.69065663e-04, 7.30665271e-01])
    )


def test_time_dependent_prob_multitask_sampler_handles_max_steps():
    sampler_1 = task_sampler.TimeDependentProbMultiTaskSampler(
        task_dict={"rte": None}, rng=0, task_to_unnormalized_prob_funcs_dict={"rte": "1"},
    )
    sampler_2 = task_sampler.TimeDependentProbMultiTaskSampler(
        task_dict={"rte": None},
        rng=0,
        task_to_unnormalized_prob_funcs_dict={"rte": "1"},
        max_steps=10,
    )
    for i in range(10):
        sampler_1.pop()
        sampler_2.pop()
    sampler_1.pop()
    with pytest.raises(IndexError):
        sampler_2.pop()
    sampler_2.reset_counter()
    sampler_2.pop()
