#!/bin/bash
#$ -cwd
#$ -j y
#$ -N randinit2-train
#$ -o ~/log/naacl/train/randinit2-train.log
#$ -l 'hostname=b1[12345678]*|c*,gpu=1, mem_free=15g, ram_free=15g'
#$ -q g.q

source activate jiant
source path_config_naacl.sh

export NFS_PROJECT_PREFIX=${TRAIN_DIR}
export JIANT_PROJECT_PREFIX=${NFS_PROJECT_PREFIX}
export CUDA_NO=`free-gpu`

EXP_NAME=randinit2
RUN_NAME=randinit2-train

OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", elmo_chars_only=1, dropout=0.2, cuda=${CUDA_NO}"
OVERRIDES+=', pretrain_tasks="none", target_tasks=mnli, random_seed=3579'
OVERRIDES+=", reload_vocab=0, reload_tasks=0, training_data_fraction=1, do_target_task_training=1, do_full_eval=1"
OVERRIDES+=", load_model=1"

python main.py --config_file config/final.conf config/naacl_additional.conf -o "${OVERRIDES}"

