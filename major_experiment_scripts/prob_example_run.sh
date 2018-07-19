#!/bin/bash

MODEL_DIR=$1 # directory of checkpoint to probe, e.g: /nfs/jsalt/share/models_to_probe/nli_do2_noelmo
PROBING_TASK=$2 # task name, e.g. recast-puns
RUN_NAME=${3:-"test"}

EXP_NAME="probing"
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"

PROB_PATH="/nfs/jsalt/home/nkim/PPProb/dat_order/word_permute_both.dev"

python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, run_name = ${RUN_NAME}, eval_tasks = ${PROBING_TASK}, ${PROBING_TASK}_use_classifier=mnli, nli-prob_path=${PROB_PATH}"
