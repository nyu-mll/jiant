#!/bin/bash
# TODO config files

SCRATCH_PREFIX='/misc/vlgscratch4/BowmanGroup/awang/'
EXP_NAME='sts'
GPUID=${2:-3}

DATE="$(date +%m-%d)"
LOG_PATH="${SCRATCH_PREFIX}/ckpts/$DATE/${EXP_NAME}/info.log"
SAVE_DIR="${SCRATCH_PREFIX}/ckpts/${DATE}/${EXP_NAME}/"
VOCAB_DIR="${SCRATCH_PREFIX}/ckpts/${DATE}/${EXP_NAME}/vocab/"
mkdir -p $SAVE_DIR
mkdir -p $VOCAB_DIR

LOAD_MODEL=${3:-1}
LOAD_TASKS=${4:-1}
LOAD_VOCAB=${5:-1}

TASKS=$1
WORD_EMBS_FILE="${SCRATCH_PREFIX}/raw_data/GloVe/glove.6B.300d.txt"

BATCH_SIZE=64
N_EPOCHS=10
LR=.1

CMD="python codebase/main.py --cuda ${GPUID} --log_file ${LOG_PATH}/${EXP_NAME}.log --tasks ${TASKS} --word_embs_file ${WORD_EMBS_FILE} --batch_size ${BATCH_SIZE} --lr ${LR}"
ALLEN_CMD="python codebase/main_allen.py --cuda ${GPUID} --log_file ${LOG_PATH} --save_dir ${SAVE_DIR} --tasks ${TASKS} --vocab_path ${VOCAB_DIR} --word_embs_file ${WORD_EMBS_FILE} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} --load_model ${LOAD_MODEL} --load_tasks ${LOAD_TASKS} --load_vocab ${LOAD_VOCAB}"
eval ${ALLEN_CMD}
#gdb --args ${ALLEN_CMD}
