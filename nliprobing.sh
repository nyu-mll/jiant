#!/bin/bash
#$ -cwd
#$ -j y
#$ -N prepswap-eval-randinit3
#$ -o ~/log/naacl/probe/prepswap-eval-randinit3.log
#$ -l 'hostname=b1[12345678]*|c*,gpu=1, mem_free=10G, ram_free=10G'
#$ -q g.q

source activate jiant
source path_config_naacl.sh

export NFS_PROJECT_PREFIX=${PROBE_DIR}
export JIANT_PROJECT_PREFIX=${NFS_PROJECT_PREFIX}
export CUDA_NO=`free-gpu`

PROBING_TASK=nli-prob-prepswap
EXP_NAME=prepswap
RUN_NAME=randinit3

MODEL_DIR=${TRAIN_DIR}/nliprobing-mlp-train/randinit3
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}"
OVERRIDES+=", use_classifier = mnli, is_probing_task=1, eval_data_fraction=1"
OVERRIDES+=", cuda = ${CUDA_NO}, load_model=1, do_target_task_training=0, reload_vocab=1"

python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf -o "${OVERRIDES}"

