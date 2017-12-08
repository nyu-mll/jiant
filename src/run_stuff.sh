#!/bin/bash
# 
#SBATCH -t 2-00:00
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=aw3272@nyu.edu

# TODO config files
SCRATCH_PREFIX='/misc/vlgscratch4/BowmanGroup/awang/'
EXP_NAME=${3:-'debug'}
GPUID=${2:-3}

DATE=${8:-"$(date +%m-%d)"}
LOG_PATH="${SCRATCH_PREFIX}/ckpts/$DATE/${EXP_NAME}/log.log"
EXP_DIR="${SCRATCH_PREFIX}/ckpts/${DATE}/${EXP_NAME}/"
VOCAB_DIR="${SCRATCH_PREFIX}/ckpts/${DATE}/${EXP_NAME}/vocab/"
mkdir -p $EXP_DIR
mkdir -p $VOCAB_DIR

LOAD_MODEL=${4:-0}
LOAD_TASKS=${5:-1}
LOAD_VOCAB=${6:-1}
LOAD_INDEX=${7:-1}

TASKS=$1
CLASSIFIER=log_reg
VOCAB_SIZE=30000
WORD_EMBS_FILE="${SCRATCH_PREFIX}/raw_data/GloVe/glove.840B.300d.txt"

N_CHAR_FILTERS=100
CHAR_FILTER_SIZES=5 #2,3,4,5
CHAR_DIM=100
WORD_DIM=300
HID_DIM=${13:-512}

PAIR_ENC=simple
N_LAYERS=2
N_HIGHWAY_LAYERS=2

OPTIMIZER=sgd
LR=.01
LR_DECAY=.2
SCHED_THRESH=1e-3
BATCH_SIZE=64
VAL_METRIC=${9:-micro}
BPP_METHOD=${10:-fixed}
BPP_BASE=${11:-10}
VAL_INTERVAL=${12:-1}
MAX_VALS=100
N_EPOCHS=10
TASK_ORDERING=${14:-small_to_large}

CMD="python codebase/main.py --cuda ${GPUID} --log_file ${LOG_PATH}/${EXP_NAME}.log --tasks ${TASKS} --word_embs_file ${WORD_EMBS_FILE} --batch_size ${BATCH_SIZE} --lr ${LR}"
ALLEN_CMD="python codebase/main_allen.py --cuda ${GPUID} --exp_name ${EXP_NAME} --log_file ${LOG_PATH} --exp_dir ${EXP_DIR} --tasks ${TASKS} --classifier ${CLASSIFIER} --vocab_path ${VOCAB_DIR} --max_vocab_size ${VOCAB_SIZE} --word_embs_file ${WORD_EMBS_FILE} --n_char_filters ${N_CHAR_FILTERS} --char_filter_sizes ${CHAR_FILTER_SIZES} --char_dim ${CHAR_DIM} --word_dim ${WORD_DIM} --hid_dim ${HID_DIM} --n_layers ${N_LAYERS} --pair_enc ${PAIR_ENC} --n_highway_layers ${N_HIGHWAY_LAYERS} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --bpp_method ${BPP_METHOD} --bpp_base ${BPP_BASE} --optimizer ${OPTIMIZER} --lr ${LR} --lr_decay_factor ${LR_DECAY} --val_interval ${VAL_INTERVAL} --max_vals ${MAX_VALS} --task_ordering ${TASK_ORDERING} --val_metric ${VAL_METRIC} --scheduler_threshold ${SCHED_THRESH} --load_model ${LOAD_MODEL} --load_tasks ${LOAD_TASKS} --load_vocab ${LOAD_VOCAB} --load_index ${LOAD_INDEX}"
eval ${ALLEN_CMD}
