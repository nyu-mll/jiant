#!/bin/bash

CLASSIFIER_TASK=mnli
EXP_NAME=nli-classifier     # experiment name
RUN_NAME=train              # run name

MODEL_DIR=""                # provide path to pretrained model dir
MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_77.best_macro.th"   # provide name of pretrained model file
PARAM_FILE=${MODEL_DIR}"/params.conf"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${CLASSIFIER_TASK}"
OVERRIDES+=", use_classifier = ${CLASSIFIER_TASK}, classifier=mlp, eval_data_fraction=1, is_probing_task=0"
OVERRIDES+=", cuda = ${CUDA_NO}, load_model=1, reload_vocab=1, reload_tasks=1, do_target_task_training=1"
OVERRIDES+=", elmo=1, elmo_chars_only=1"
#OVERRIDES+=", sent_enc=bilm"   # uncomment for models pretrained with language modeling

python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf config/naacl_additional.conf -o "${OVERRIDES}" 

