import json
import os

import torch

import jiant.utils.python.io as py_io
import jiant.proj.main.components.task_sampler as jiant_task_sampler


def write_val_results(val_results_dict, metrics_aggregator, output_dir, verbose=True):
    full_results_to_write = {
        "aggregated": jiant_task_sampler.compute_aggregate_major_metrics_from_results_dict(
            metrics_aggregator=metrics_aggregator, results_dict=val_results_dict,
        ),
    }
    for task_name, task_results in val_results_dict.items():
        task_results_to_write = {}
        if "loss" in task_results:
            task_results_to_write["loss"] = task_results["loss"]
        if "metrics" in task_results:
            task_results_to_write["metrics"] = task_results["metrics"].to_dict()
        full_results_to_write[task_name] = task_results_to_write

    metrics_str = json.dumps(full_results_to_write, indent=2)
    if verbose:
        print(metrics_str)

    py_io.write_json(data=full_results_to_write, path=os.path.join(output_dir, "val_metrics.json"))


def write_preds(eval_results_dict, path):
    preds_dict = {}
    for task_name, task_results_dict in eval_results_dict.items():
        preds_dict[task_name] = {
            "preds": task_results_dict["preds"],
            "guids": task_results_dict["accumulator"].get_guids(),
        }
    torch.save(preds_dict, path)
