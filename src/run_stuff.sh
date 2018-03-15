#!/bin/bash
# 
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mail-type=end
#SBATCH --mail-user=aw3272@nyu.edu

# SBATCH -t 4-00:00
# SBATCH --gres=gpu:p40:1

# TODO config files
SCRATCH_PREFIX='/misc/vlgscratch4/BowmanGroup/awang/'
#SCRATCH_PREFIX='/beegfs/aw3272/'
PROJECT_NAME='mtl-sent-rep'
EXP_NAME="debug"
RUN_NAME="debug"
GPUID=0
RANDOM_SEED=19

SHOULD_TRAIN=1
LOAD_MODEL=0
LOAD_TASKS=1
LOAD_VOCAB=1
LOAD_INDEX=1

TASKS='all'
CLASSIFIER=log_reg
VOCAB_SIZE=100000
CHAR_VOCAB_SIZE=100
WORD_EMBS_FILE="${SCRATCH_PREFIX}/raw_data/GloVe/glove.840B.300d.txt"

N_CHAR_FILTERS=100
CHAR_FILTER_SIZES=5
CHAR_DIM=100
WORD_DIM=300
HID_DIM=512
ELMO=0
COVE=0

PAIR_ENC="simple"
N_LAYERS=2
N_HIGHWAY_LAYERS=2

OPTIMIZER="sgd"
LR=1.
LR_DECAY=.5
WEIGHT_DECAY=0.0
SCHED_THRESH=1e-3
BATCH_SIZE=64
BPP_METHOD="percent_tr"
BPP_BASE=1
VAL_INTERVAL=1
MAX_VALS=100
N_EPOCHS=10
TASK_ORDERING="small_to_large"

while getopts 'ikmn:r:s:tvO:h:l:L:e:o:T:b:H:p:Ec' flag; do
    case "${flag}" in
        n) EXP_NAME="${OPTARG}" ;;
        r) RUN_NAME="${OPTARG}" ;;
        s) SEED="${OPTARG}" ;;
        t) SHOULD_TRAIN=0 ;;
        v) LOAD_VOCAB=0 ;;
        k) LOAD_TASKS=0 ;;
        m) LOAD_MODEL=1 ;;
        i) LOAD_INDEX=0 ;;
        T) TASKS="${OPTARG}" ;;
        O) TASK_ORDERING="${OPTARG}" ;;
        H) N_HIGHWAY_LAYERS="${OPTARG}" ;;
        l) LR="${OPTARG}" ;;
        L) N_LAYERS="${OPTARG}" ;;
        e) PAIR_ENC="${OPTARG}" ;;
        o) OPTIMIZER="${OPTARG}" ;;
        h) HID_DIM="${OPTARG}" ;;
        b) BATCH_SIZE="${OPTARG}" ;;
        p) PAIR_ENC="${OPTARG}" ;;
        E) ELMO=1 ;;
        c) COVE=1 ;;
    esac
done

LOG_PATH="${SCRATCH_PREFIX}/ckpts/${PROJECT_NAME}/${EXP_NAME}/${RUN_NAME}/log.log"
EXP_DIR="${SCRATCH_PREFIX}/ckpts/${PROJECT_NAME}/${EXP_NAME}/${RUN_NAME}"
VOCAB_DIR="${SCRATCH_PREFIX}/ckpts/${PROJECT_NAME}/${EXP_NAME}/vocab/"
mkdir -p $EXP_DIR
mkdir -p $VOCAB_DIR

CMD="python codebase/main.py --cuda ${GPUID} --log_file ${LOG_PATH}/${EXP_NAME}.log --tasks ${TASKS} --word_embs_file ${WORD_EMBS_FILE} --batch_size ${BATCH_SIZE} --lr ${LR}"
ALLEN_CMD="python codebase/main_allen.py --cuda ${GPUID} --random_seed ${RANDOM_SEED} --exp_name ${EXP_NAME} --log_file ${LOG_PATH} --exp_dir ${EXP_DIR} --tasks ${TASKS} --classifier ${CLASSIFIER} --vocab_path ${VOCAB_DIR} --max_vocab_size ${VOCAB_SIZE} --max_char_vocab_size ${CHAR_VOCAB_SIZE} --word_embs_file ${WORD_EMBS_FILE} --elmo ${ELMO} --cove ${COVE} --n_char_filters ${N_CHAR_FILTERS} --char_filter_sizes ${CHAR_FILTER_SIZES} --char_dim ${CHAR_DIM} --word_dim ${WORD_DIM} --hid_dim ${HID_DIM} --n_layers ${N_LAYERS} --pair_enc ${PAIR_ENC} --n_highway_layers ${N_HIGHWAY_LAYERS} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --bpp_method ${BPP_METHOD} --bpp_base ${BPP_BASE} --optimizer ${OPTIMIZER} --lr ${LR} --lr_decay_factor ${LR_DECAY} --weight_decay ${WEIGHT_DECAY} --val_interval ${VAL_INTERVAL} --max_vals ${MAX_VALS} --task_ordering ${TASK_ORDERING} --scheduler_threshold ${SCHED_THRESH} --load_model ${LOAD_MODEL} --load_tasks ${LOAD_TASKS} --load_vocab ${LOAD_VOCAB} --load_index ${LOAD_INDEX} --should_train ${SHOULD_TRAIN}"
eval ${ALLEN_CMD}
