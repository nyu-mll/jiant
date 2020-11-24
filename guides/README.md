# Guides

Also check out our [Examples](../examples) to see `jiant` in action.

If you don't know what to read, why not read our [In-Depth Introduction to Jiant](general/in_depth_intro.md)?

Contents:

* [Tutorials](#tutorials)
* [General](#general)
* [Benchmarks](#benchmarks)
* [Experiments](#experiments)
* [Tasks](#tasks)
* [Papers / Projects](#papers--projects)

---

## Tutorials

These are quick tutorials that demonstrate `jiant` usage.

* [Quick Start Guide — Using the "Simple" CLI](tutorials/quick_start_simple.md): A simple `jiant` training run in bash, using the "Simple" CLI
* [Quick Start Guide — Using the "Main" CLI](tutorials/quick_start_main.md): A simple `jiant` training run in bash, using the "Main" CLI

The "Simple" API provides a single command-line script for training and evaluating models on tasks, while the "Main" API offers more flexibilty by breaking the workflow down into discrete steps (downloading the model, tokenization & caching, writing a fully specific run-configuration, and finally running the experiment). Both interfaces use the same models and task implementations uner the hood.


## General

These are general guides to `jiant`'s design and components. Refer to these if you have questions about parts of `jiant`:

* [In-Depth Introduction to Jiant](general/in_depth_intro.md): Learn about `jiant` in greater detail
    * [`jiant`'s models](general/in_depth_intro.md#jiants-models)
    * [`jiant`'s tasks](general/in_depth_intro.md#jiants-tasks)
    * [`Runner`s and `Metarunner`s](general/in_depth_intro.md#runners-and-metarunners)
    * [Step-by-step through `jiant`'s pipeline](general/in_depth_intro.md#step-by-step-through-jiants-pipeline)

## Running benchmarks

These are guides to running common NLP benchmarks using `jiant`:

* [GLUE Benchmark](benchmarks/glue.md): Generate GLUE Benchmark submissions
* [SuperGLUE Benchmark](benchmarks/superglue.md): Generate SuperGLUE Benchmark submissions
* [XTREME](benchmarks/xtreme.md): End-to-end guide for training and generating submission for the XTREME bernchmark

## Tips & Tricks for Running Experiments

These are more specific guides about running experiments in `jiant`:

* [My Experiment and Me](experiments/my_experiment_and_me.md): More info about a `jiant` training/eval run
* [Tips for Large-scale Experiments](experiments/large_scale_experiments.md)

## Tasks

These are notes on the tasks supported in `jiant`:

* [List of supported tasks in `jiant`](tasks/supported_tasks.md)
* [Task-specific notes](tasks/task_specific.md): Learn about quirks/caveats about specific tasks
* [Adding Tasks](tasks/adding_tasks.md): Guide on adding a task to `jiant`

## Papers / Projects

* [English Intermediate-Task Training Improves Zero-Shot Cross-Lingual Transfer Too (X-STILTs)](projects/xstilts.md)