#!/bin/bash
#$ -cwd
#$ -j y
#$ -N coord-turked-eval-ccg-fullelmo
#$ -o ~/log/naacl/probe/coord-turked-eval-ccg-fullelmo.log
#$ -l 'hostname=b1[12345678]*|c*,gpu=1, mem_free=10G, ram_free=10G'
#$ -q g.q

source activate jiant
source path_config_naacl.sh

export NFS_PROJECT_PREFIX=${PROBE_DIR}
export JIANT_PROJECT_PREFIX=${NFS_PROJECT_PREFIX}
export CUDA_NO=`free-gpu`

PROBING_TASK=coord-turked
EXP_NAME=coord-turked-fullelmo
RUN_NAME=ccg-fullelmo
BEST_EP=7

MODEL_DIR=${TRAIN_DIR}/${RUN_NAME}/${RUN_NAME}-train
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_${BEST_EP}.best_macro.th"

#MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_0.th"


OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}"
OVERRIDES+=", use_classifier = ${PROBING_TASK}"
OVERRIDES+=", cuda = ${CUDA_NO}, load_model=1, reload_vocab=0, do_target_task_training=1"
OVERRIDES+=", elmo=1, elmo_chars_only=0"
#OVERRIDES+=", scaling_method=uniform"

python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf config/naacl_additional.conf -o "${OVERRIDES}"

