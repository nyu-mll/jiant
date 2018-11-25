#!/bin/bash
#$ -cwd
#$ -j y
#$ -N constonto-eval-mt-elmo
#$ -o ~/log/naacl/probe/constonto-eval-mt-elmo.log
#$ -l 'hostname=b1[12345678]*|c0*|c1[2345678]*,gpu=1, mem_free=15G, ram_free=15G'
#$ -q g.q

source activate jiant
source path_config_naacl.sh

export NFS_PROJECT_PREFIX=${PROBE_DIR}
export JIANT_PROJECT_PREFIX=${NFS_PROJECT_PREFIX}
export CUDA_NO=`free-gpu`

EXP_NAME=constonto-elmo
RUN_NAME=mt-elmo
BEST_EP=539

#PROBING_TASK="edges-srl-conll2005"
#PROBING_TASK="edges-srl-conll2012"
#PROBING_TASK="edges-dep-labeling-ewt"
PROBING_TASK="edges-constituent-ontonotes"

MODEL_DIR=${TRAIN_DIR}/${RUN_NAME}/${RUN_NAME}-train
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_${BEST_EP}.best_macro.th"

# use this for random init models
#MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_0.th"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}"
OVERRIDES+=", use_classifier = ${PROBING_TASK}"
OVERRIDES+=", cuda = ${CUDA_NO}, load_model=1, reload_vocab=0, do_target_task_training=1, reload_tasks=0"
OVERRIDES+=", elmo=1, elmo_chars_only=0"
#OVERRIDES+=", scaling_method=uniform"

python main.py -c config/final.conf ${PARAM_FILE} config/edgeprobe_existing.conf config/naacl_additional.conf -o "${OVERRIDES}"

