#!/bin/bash
#$ -cwd
#$ -j y
#$ -N mlp-mt-fullelmo
#$ -o ~/log/naacl/train/mlp-mt-fullelmo.log
#$ -l 'hostname=b1[12345678]*|c*,gpu=1, ram_free=10G, mem_free=10G'
#$ -q g.q

source activate jiant
source path_config_naacl.sh

export NFS_PROJECT_PREFIX=${TRAIN_DIR}
export JIANT_PROJECT_PREFIX=${NFS_PROJECT_PREFIX}
export CUDA_NO=`free-gpu`

PROBING_TASK=mnli
EXP_NAME=nliprobing-mlp-train
RUN_NAME=mt-fullelmo

MODEL_DIR=${TRAIN_DIR}/${RUN_NAME}/${RUN_NAME}-train
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_539.best_macro.th"

# Use for random init  models
#MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_0.th"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}"
OVERRIDES+=", use_classifier = ${PROBING_TASK}, classifier=mlp, eval_data_fraction=1, is_probing_task=0"
OVERRIDES+=", cuda = ${CUDA_NO}, load_model=1, reload_vocab=0, elmo_chars_only=1, do_target_task_training=1"
OVERRIDES+=", elmo=1, elmo_chars_only=0"

python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf config/naacl_additional.conf -o "${OVERRIDES}" 

