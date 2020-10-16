# English Intermediate-Task Training Improves Zero-Shot Cross-Lingual Transfer Too

This guide describes how to replicate the experiments in [English Intermediate-Task Training Improves Zero-Shot Cross-Lingual Transfer Too](https://arxiv.org/abs/2005.13013).

## Overview

The experiments described in the paper follow a simple transfer learning procedure: 

1. Fine-tune XLM-R (large) on an intermediate task (e.g. MNLI, SQuAD), or on multiple tasks
2. Evaluate the checkpoint from (1) on the XTREME benchmark.

In the second step, this entails tuning the checkpoint from (1) on the English training sets of the 7 XTREME tasks, and zero-shot evaluation on the tuned models on the same tasks in other languages. There are also 2 XTREME tasks where the checkpoint from (1) is evaluated directly (BuCC, Tatoeba).

## Intermediate-Task Training

### Single/Multiple Intermediate Task

For tuning on a single or multiple intermediate tasks, you can use the [Quick Start guide](../tutorials/quick_start_simple.md) as a reference. You should follow the same steps for downloading the data. Be sure to use the relevant tasks and XLM-R models. For instance, the training command should look something like:

```bash
python jiant/jiant/proj/simple/runscript.py \
    run \
    --run_name mnli_and_squad \
    --exp_dir ./experiments/stilts \
    --data_dir $(pwd)/tasks/data \
    --model_type xlm-roberta-large \
    --train_batch_size 4 \
    --tasks mnli,squad_v1
``` 

## XTREME Benchmark Evaluation

The [XTREME benchmark guide](../benchmarks/xtreme.md) describes how to evaluate XLM-R on the XTREME benchmark, end-to-end. You can generally follow the guide, except two steps:

1. You don't need to re-download `xlm-roberta-large`, since you should have a checkpoint from the previous step.
2. In the [final step](../benchmarks/xtreme.md#trainrun-models) where you train/run the models, replace the following line
```bash
        --model_load_mode from_transformers \
```
with
```bash
        --ZZoverrides model_load_path \
        --model_load_mode partial \
        --model_load_path /path/to/my/model.p \
```

This ensures that your encoder is loaded from the model tuned on the intermediate task.

## Citation

If you would like to cite our work:

Jason Phang, Iacer Calixto, Phu Mon Htut, Yada Pruksachatkun, Haokun Liu, Clara Vania, Katharina Kann, and Samuel R. Bowman  **"English Intermediate-Task Training Improves Zero-Shot Cross-Lingual Transfer Too."** *Proceedings of AACL, 2020*

```
@inproceedings{phang2020english,
    author = {Jason Phang and Iacer Calixto and Phu Mon Htut and Yada Pruksachatkun and Haokun Liu and Clara Vania and Katharina Kann and Samuel R. Bowman},
    title = {English Intermediate-Task Training Improves Zero-Shot Cross-Lingual Transfer Too},
    booktitle = {Proceedings of AACL},
    year = {2020}
}
```
