#!/bin/bash

PARAM_FILE="/nfs/jsalt/exp/katherin-worker6/main-wmt/noelmo/params.conf"
MODEL_FILE="/nfs/jsalt/exp/katherin-worker6/main-wmt/noelmo/model_state_main_epoch_49.best_macro.th"

EXP_NAME="main-wmt"
RUN_NAME="noelmo_eval"


python main.py --c config/defaults.conf,${PARAM_FILE},config/eval_existing.conf -o "load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, run_name = ${RUN_NAME}, eval_tasks = glue"
