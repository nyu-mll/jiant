# Quick Start Guide — Using the "Simple" CLI

In this tutorial we'll show you how to do a basic training experiment using `jiant`'s command line interface.

We will assume that `jiant` and its dependencies have already be installed.

## Workflow

First, let us assume that we will be working with the following working directory:
```
EXP_DIR=/path/to/exp
```

For this training example we'll use the RTE task from GLUE, and the RoBERTa-base model. 

1. We'll get the data using `jiant`'s download script
```
python jiant/scripts/download_data/runscript.py \
    download \
    --tasks rte \
    --output_path ${EXP_DIR}/tasks
```

2. Now that the data is ready, we can use `jiant`'s "Simple" CLI to perform training with a single command:
```
python jiant/proj/simple/runscript.py \
    run \
    --run_name simple \
    --exp_dir ${EXP_DIR} \
    --data_dir ${EXP_DIR}/tasks \
    --model_type roberta-base \
    --tasks rte \
    --train_batch_size 16 \
    --num_train_epochs 3 \
    --do_save
```

The "Simple" CLI subsumes several steps under the hood, including downloading the `roberta-base` model, tokenizing and caching the data, writing a [run-configuration](../general/in_depth_into.md#write-run-config), and performing the training. 
