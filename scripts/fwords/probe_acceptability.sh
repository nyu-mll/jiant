#!/bin/bash

PROBING_TASK=acceptability-conj     # acceptability-{conj, def, eos, wh}
FOLD_NO=1                           # xval fold number
EXP_NAME=dissent                    # experiment name
RUN_NAME=probe                      # run name

MODEL_DIR=""                        # directory where the pretrained model files are located
MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_26.best_macro.th"   # name of the pretrained model file to be probed
PARAM_FILE=${MODEL_DIR}"/params.conf"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}, fold_no=${FOLD_NO}"
OVERRIDES+=", cuda = ${CUDA_NO}, load_model=1, reload_vocab=1, reload_tasks=1, do_target_task_training=1, is_probing_task=0"
OVERRIDES+=", elmo=1, elmo_chars_only=1"

#OVERRIDES+=", sent_enc=bilm"       # uncomment for models trained on language modeling

python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf config/naacl_additional.conf -o "${OVERRIDES}"
