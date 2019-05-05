#!/bin/bash

PROBING_TASK=nli-prob-negation      # nli-prob-{negation, prep, spatial, quant, comp}
EXP_NAME=mnli-pretrained
RUN_NAME=probe

MODEL_DIR=""                        # path to trained NLI classifier directory
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th" # Do not modify

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}"
OVERRIDES+=", use_classifier = mnli, is_probing_task=1, eval_data_fraction=1"
OVERRIDES+=", cuda = ${CUDA_NO}, load_model=1, do_target_task_training=0, reload_vocab=1, reload_tasks=1"
OVERRIDES+=", elmo=1, elmo_chars_only=1, write_preds=val"
OVERRIDES+=", ${PROBING_TASK}_use_classifier=mnli"
#OVERRIDES+=", sent_enc=bilm"       # uncomment for models pretrained on language modeling

python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf config/naacl_additional.conf -o "${OVERRIDES}"

