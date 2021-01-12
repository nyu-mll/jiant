# Quick Start Guide — Using the "Main" CLI

In this tutorial we'll show you how to do a basic multitask training experiment using `jiant`'s command line interface.

We will assume that `jiant` and its dependencies have already be installed.

## Workflow

First, let us assume that we will be working with the following working directory. For this training example we'll use the RTE task from GLUE, and the RoBERTa-base model.
```
EXP_DIR=/path/to/exp
MODEL_TYPE=roberta-base
TASK=rte
```

Neither of these environment variables are read directly in Python, but simply used to make our bash commands more succinct. This also writes the intermediate and output files in our recommended folder organization.

1. We'll get the data using `jiant`'s download script
```bash
python jiant/scripts/download_data/runscript.py \
    download \
    --tasks ${TASK} \
    --output_path ${EXP_DIR}/tasks/
```

2. Next, we download our RoBERTa-base model
```bash
python jiant/proj/main/export_model.py \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_base_path ${EXP_DIR}/models/${MODEL_TYPE}
```

3. Next, we tokenize and cache the inputs and labels for our RTE task
```bash
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${EXP_DIR}/tasks/configs/${TASK}_config.json \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_dir ${EXP_DIR}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val \
    --max_seq_length 256 \
    --smart_truncate
```

4. Next, we write a [run-config](../general/in_depth_intro.md#write-run-config). This writes a JSON file that specifies some configuration for our run - it's pretty simple now, but will be much more meaningful when running more complex multitasking experiments.
```bash
python jiant/proj/main/scripts/configurator.py \
    SingleTaskConfigurator \
    ${EXP_DIR}/runconfigs/${MODEL_TYPE}/${TASK}.json \
    --task_name ${TASK} \
    --task_config_base_path ${EXP_DIR}/tasks/configs \
    --task_cache_base_path ${EXP_DIR}/cache/${MODEL_TYPE} \
    --epochs 3 \
    --train_batch_size 16 \
    --eval_batch_multiplier 2 \
    --do_train --do_val
```

5. Finally, we train our model.
```bash
python jiant/proj/main/runscript.py \
    run \
    --ZZsrc ${EXP_DIR}/models/${MODEL_TYPE}/config.json \
    --jiant_task_container_config_path ${EXP_DIR}/runconfigs/${MODEL_TYPE}/${TASK}.json \
    --model_load_mode from_transformers \
    --learning_rate 1e-5 \
    --do_train --do_val \
    --do_save --force_overwrite \
    --output_dir ${EXP_DIR}/runs/${MODEL_TYPE}/${TASK}
```


## Additional Options

### Saving model weights

To save the model weights, use the `--do_save` flag:

```bash
python jiant/proj/simple/runscript.py \
    ...
    --do_save
```

This will save two sets of model weights: `last_model.p` will the the model weights at the end of training, while `best_model.p` will be the weights which achieves the best validation score on the evaluation subset (see: [Early Stopping](#early-stopping)). If you only one to save one or the other, use `--do_save_last` or `--do_save_best`.

### Checkpointing

To allow checkpointing (allowing you to resume runs that get interrupted), use the `run_with_continue` mode and set the `save_checkpoint_every_steps` argument. For example:

```bash
python jiant/proj/main/runscript.py \
    run_with_continue \
        ...
        ...
    --save_checkpoint_every_steps 500 \
    --delete_checkpoint_if_done
```

This will save a checkpoint to disk every 500 training steps. The checkpoint will be saved to a `checkpoint.p` file. If the process gets killed, you can rerun the exact same command and it will continue training from the latest checkpoint.

Note that checkpoints are for resuming training, not for saving snapshots of model weights at different points in training. Checkpoints also include additional run metadata, as well as the optimizer states. To save regular snapshots of model weights, see [Model Snapshots](#model-snapshots)

We also set the `delete_checkpoint_if_done` flag to delete the checkpoint after training is complete.

### Model Snapshots

To save snapshots of model weights at regular intervals, use the `--save_every_steps` argument. For example:

```
    --save_every_steps 500
```

will save a pickle of model weights every 500 training steps.

### Early Stopping

To do early stopping, we can perform validation evaluation at regular intervals over the course of training, and select the best model weights based on validation performance. For expedience, we often do not want to evaluate on the whole validation set, but only a subset. To do early stopping, use the following arguments as an example:

```
    --eval_every_steps 1000
    --no_improvements_for_n_evals 30
    --eval_subset_num 500
```

* `--eval_every_steps 1000` indicates that we will evaluate the model on a validation subset every 1000 training steps.
* `--no_improvements_for_n_evals 30` indicates that if the validation performance does not improve for 30 consecutive validation evaluations, we will end the training phase
* `--eval_subset_num 500` indicates that we will evaluate on the first 500 validation examples for early stopping. This value is `500` by default.
