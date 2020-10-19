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
    --model_type ${MODEL_TYPE} \
    --output_base_path ${EXP_DIR}/models/${MODEL_TYPE}
```

3. Next, we tokenize and cache the inputs and labels for our RTE task
```bash
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${EXP_DIR}/tasks/configs/${TASK}_config.json \
    --model_type ${MODEL_TYPE} \
    --model_tokenizer_path \
    ${EXP_DIR}/models/${MODEL_TYPE}/tokenizer \
    --output_dir ${EXP_DIR}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val \
    --max_seq_length 256 \
    --smart_truncate
```  

4. Next, we write a [run-config](../general/in_depth_into.md#write-run-config). This writes a JSON file that specifies some configuration for our run - it's pretty simple now, but will be much more meaningful when running more complex multitasking experiments. 
```bash
python jiant/proj/main/scripts/configurator.py \
    SingleTaskConfigurator \
    ${EXP_DIR}/runconfigs/${MODEL_TYPE}/${TASK}.json \
    --task_name ${TASK} \
    --task_config_base_path ${EXP_DIR}/tasks/configs \
    --task_cache_base_path ${EXP_DIR}/cache/${MODEL_TYPE} \
    --epochs 3 \
    --train_batch_size 4 \
    --eval_batch_multiplier 2 \
    --do_train --do_val
```

5. Finally, we train our model.
```bash
python jiant/proj/main/runscript.py \
    run_with_continue \
    --ZZsrc ${EXP_DIR}/models/${MODEL_TYPE}/config.json \
    --jiant_task_container_config_path ${EXP_DIR}/runconfigs/${MODEL_TYPE}/${TASK}.json \
    --model_load_mode from_transformers \
    --learning_rate 1e-5 \
    --eval_every_steps 1000 \
    --no_improvements_for_n_evals 30 \
    --save_checkpoint_every_steps 1000 \
    --delete_checkpoint_if_done \
    --do_save --do_train --do_val \
    --force_overwrite \
    --output_dir ${EXP_DIR}/runs/${MODEL_TYPE}/${TASK}
```