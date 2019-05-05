#!/bin/bash

PRETRAIN_TASK=ccg   # name of pretraining task as defined by @register_task in src/tasks.py
EXP_NAME=ccg        # experiment name
RUN_NAME=ccg-train  # run name

OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", pretrain_tasks=${PRETRAIN_TASK}"
OVERRIDES+=', target_tasks=none'
OVERRIDES+=", elmo_chars_only=1, dropout=0.2, random_seed=1234, cuda=${CUDA_NO}, reload_vocab=0, reload_tasks=0, training_data_fraction=1, do_target_task_training=0, do_full_eval=0, load_model=1"
#OVERRIDES+=", sent_enc=bilm"  # uncomment for language modeling pretraining

python main.py --config_file config/final.conf config/naacl_additional.conf -o "${OVERRIDES}"

