#!/bin/bash

PARAM_FILE=$1 #"/nfs/jsalt/exp/ellie-newworker-1/bie/ptdb_lowdropout_noelmo/params.conf"
MODEL_FILE=$2 #"/nfs/jsalt/exp/ellie-newworker-1/bie/ptdb_lowdropout_noelmo/model_state_main_epoch_5.best_macro.th"
PROBING_TASK=$3 #"recast-puns"

EXP_NAME="probing"
RUN_NAME=$4 #"ptdb_pun_test"

echo '"load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, run_name = ${RUN_NAME}, eval_tasks = ${PROBING_TASK} ${PROBING_TASK}_use_classifier=mnli"'

python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, run_name = ${RUN_NAME}, eval_tasks = ${PROBING_TASK} ${PROBING_TASK}_use_classifier=mnli"
