#!/bin/bash
# 
#SBATCH -t 2-00:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH -o /misc/vlgscratch4/BowmanGroup/awang/ckpts/sbatch.out
#SBATCH -e /misc/vlgscratch4/BowmanGroup/awang/ckpts/sbatch.err
#SBATCH --mail-type=end
#SBATCH --mail-user=aw3272@nyu.edu

# TODO config files

SCRATCH_PREFIX='/misc/vlgscratch4/BowmanGroup/awang/'
EXP_NAME=${3:-'debug'}
GPUID=${2:-3}

###
# TODO: UNDO THIS FIXED VALUE
##

DATE="11-28" #"$(date +%m-%d)"
LOG_PATH="${SCRATCH_PREFIX}/ckpts/$DATE/${EXP_NAME}/info.log"
EXP_DIR="${SCRATCH_PREFIX}/ckpts/${DATE}/${EXP_NAME}/"
VOCAB_DIR="${SCRATCH_PREFIX}/ckpts/${DATE}/${EXP_NAME}/vocab/"
mkdir -p $EXP_DIR
mkdir -p $VOCAB_DIR

LOAD_MODEL=${4:-1}
LOAD_TASKS=${5:-1}
LOAD_VOCAB=${6:-1}
REINDEX=${7:-1}

TASKS=$1
CLASSIFIER=log_reg
VOCAB_SIZE=30000
WORD_EMBS_FILE="${SCRATCH_PREFIX}/raw_data/GloVe/glove.840B.300d.txt"

N_CHAR_FILTERS=100
CHAR_FILTER_SIZES=5 #2,3,4,5
CHAR_DIM=100
WORD_DIM=300
HID_DIM=${8:-2048}

PAIR_ENC=simple
N_LAYERS=2
N_HIGHWAY_LAYERS=2

BATCH_SIZE=64
N_EPOCHS=10
OPTIMIZER=sgd
LR=1.

N_PASSES=1
TASK_ORDERING='small_to_large'

CMD="python codebase/main.py --cuda ${GPUID} --log_file ${LOG_PATH}/${EXP_NAME}.log --tasks ${TASKS} --word_embs_file ${WORD_EMBS_FILE} --batch_size ${BATCH_SIZE} --lr ${LR}"
ALLEN_CMD="python codebase/main_allen.py --cuda ${GPUID} --exp_name ${EXP_NAME} --log_file ${LOG_PATH} --exp_dir ${EXP_DIR} --tasks ${TASKS} --classifier ${CLASSIFIER} --vocab_path ${VOCAB_DIR} --max_vocab_size ${VOCAB_SIZE} --word_embs_file ${WORD_EMBS_FILE} --n_char_filters ${N_CHAR_FILTERS} --char_filter_sizes ${CHAR_FILTER_SIZES} --char_dim ${CHAR_DIM} --word_dim ${WORD_DIM} --hid_dim ${HID_DIM} --n_layers ${N_LAYERS} --pair_enc ${PAIR_ENC} --n_highway_layers ${N_HIGHWAY_LAYERS} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --optimizer ${OPTIMIZER} --lr ${LR} --n_passes_per_epoch ${N_PASSES} --task_ordering ${TASK_ORDERING} --load_model ${LOAD_MODEL} --load_tasks ${LOAD_TASKS} --load_vocab ${LOAD_VOCAB} --reindex ${REINDEX}"
eval ${ALLEN_CMD}
#gdb --args ${ALLEN_CMD}
