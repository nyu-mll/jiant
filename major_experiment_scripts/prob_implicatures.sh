#!/bin/bash

MODEL_DIR=$1 # directory of checkpoint to probe, e.g: /nfs/jsalt/share/models_to_probe/nli_do2_noelmo
PROBING_TASK=$2 # task name, e.g. recast-puns
RUN_NAME=${3:-"test"}
PROBE_VERB=$4

PROBE_PATH="/nfs/jsalt/exp/alexis-probing/data/implicatures/"${PROBE_VERB}"/indexed_TEST.tsv"

EXP_NAME=${PROBE_VERB}
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"

python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, run_name = ${RUN_NAME}, nli-prob {probe_path = ${PROBE_PATH}}, eval_tasks = ${PROBING_TASK}, ${PROBING_TASK}_use_classifier=mnli"
