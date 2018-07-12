#!/bin/bash

PARAM_FILE="/nfs/jsalt/exp/ellie-newworker-1/bie/ptdb_lowdropout_noelmo/params.conf"
MODEL_FILE="/nfs/jsalt/exp/ellie-newworker-1/bie/ptdb_lowdropout_noelmo/model_state_main_epoch_5.best_macro.th"
PROBING_TASK="recast-puns"

EXP_NAME="probing_test"
RUN_NAME="ptdb_pun_test"


python main.py --c config/defaults.conf,${PARAM_FILE},config/eval_existing.conf -o "load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, run_name = ${RUN_NAME}, eval_tasks = ${PROBING_TASK}"
