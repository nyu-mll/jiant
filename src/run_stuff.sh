#!/bin/bash
# TODO config files

SCRATCH_PREFIX='/misc/vlgscratch4/BowmanGroup/awang/'
EXP_NAME='toy'
GPUID=${1:-3}

DATE="$(date +%m-%d)"
LOG_PATH="logs/$DATE/"
mkdir -p $LOG_PATH

TASKS=snli
WORD_EMBS_FILE="${SCRATCH_PREFIX}/raw_data/GloVe/glove.6B.300d.txt"

BATCH_SIZE=128
LR=.1

CMD="python codebase/main.py --cuda ${GPUID} --log_file ${LOG_PATH}/${EXP_NAME}.log --tasks ${TASKS} --word_embs_file ${WORD_EMBS_FILE} --batch_size ${BATCH_SIZE} --lr ${LR}"
eval $CMD
