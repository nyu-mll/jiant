#!/bin/bash
# 
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mail-type=end
#SBATCH --mail-user=aw3272@nyu.edu

# SBATCH -t 4-00:00
# SBATCH --gres=gpu:p40:1

#SCRATCH_PREFIX='/misc/vlgscratch4/BowmanGroup/awang/'
SCRATCH_PREFIX='/beegfs/aw3272/'
PROJECT_NAME='mtl-sent-rep'
EXP_NAME="debug"
RUN_NAME="debug"
GPUID=0
SEED=19

SHOULD_TRAIN=1
LOAD_MODEL=0
LOAD_TASKS=1
LOAD_PREPROC=1

train_tasks='all'
eval_tasks='none'
CLASSIFIER=mlp
d_hid_cls=512
VOCAB_SIZE=100000
CHAR_VOCAB_SIZE=100
WORD_EMBS_FILE="${SCRATCH_PREFIX}/raw_data/GloVe/glove.840B.300d.txt"

N_CHAR_FILTERS=100
CHAR_FILTER_SIZES=5
d_char=100
d_word=300
d_hid=512
ELMO=0
deep_elmo=0
elmo_no_glove=0
COVE=0

PAIR_ENC="simple"
N_LAYERS_ENC=2
n_layers_highway=2

OPTIMIZER="sgd"
LR=.1
min_lr=1e-5
LR_DECAY=.2
train_words=0
WEIGHT_DECAY=0.0
SCHED_THRESH=0.0
BATCH_SIZE=64
BPP_METHOD="percent_tr"
BPP_BASE=1
VAL_INTERVAL=1
MAX_VALS=100
N_EPOCHS=10
TASK_ORDERING="small_to_large"

while getopts 'ikmn:r:S:s:tvh:l:L:o:T:E:O:b:H:p:edcgP:' flag; do
    case "${flag}" in
        P) SCRATCH_PREFIX="${OPTARG}" ;;
        n) EXP_NAME="${OPTARG}" ;;
        r) RUN_NAME="${OPTARG}" ;;
        S) SEED="${OPTARG}" ;;
        t) SHOULD_TRAIN=0 ;;
        k) LOAD_TASKS=0 ;;
        m) LOAD_MODEL=0 ;;
        i) LOAD_PREPROC=0 ;;
        T) train_tasks="${OPTARG}" ;;
        E) eval_tasks="${OPTARG}" ;;
        O) TASK_ORDERING="${OPTARG}" ;;
        H) n_layers_highway="${OPTARG}" ;;
        l) LR="${OPTARG}" ;;
        s) min_lr="${OPTARG}" ;;
        L) N_LAYERS_ENC="${OPTARG}" ;;
        o) OPTIMIZER="${OPTARG}" ;;
        h) d_hid="${OPTARG}" ;;
        b) BATCH_SIZE="${OPTARG}" ;;
        p) PAIR_ENC="${OPTARG}" ;;
        e) ELMO=1 ;;
        d) deep_elmo=1 ;;
        g) elmo_no_glove=1 ;;
        c) COVE=1 ;;
    esac
done

LOG_PATH="${SCRATCH_PREFIX}/ckpts/${PROJECT_NAME}/${EXP_NAME}/${RUN_NAME}/log.log"
EXP_DIR="${SCRATCH_PREFIX}/ckpts/${PROJECT_NAME}/${EXP_NAME}/${RUN_NAME}"
PREPROC_FILE="${EXP_DIR}/preproc.pkl"
VOCAB_DIR="${SCRATCH_PREFIX}/ckpts/${PROJECT_NAME}/${EXP_NAME}/vocab/"
mkdir -p $EXP_DIR
mkdir -p $VOCAB_DIR

ALLEN_CMD="python main.py --cuda ${GPUID} --random_seed ${SEED} --log_file ${LOG_PATH} --exp_dir ${EXP_DIR} --train_tasks ${train_tasks} --eval_tasks ${eval_tasks} --classifier ${CLASSIFIER} --classifier_hid_dim ${d_hid_cls} --vocab_path ${VOCAB_DIR} --max_word_v_size ${VOCAB_SIZE} --max_char_v_size ${CHAR_VOCAB_SIZE} --word_embs_file ${WORD_EMBS_FILE} --train_words ${train_words} --elmo ${ELMO} --deep_elmo ${deep_elmo} --elmo_no_glove ${elmo_no_glove} --cove ${COVE} --n_char_filters ${N_CHAR_FILTERS} --char_filter_sizes ${CHAR_FILTER_SIZES} --d_char ${d_char} --d_word ${d_word} --d_hid ${d_hid} --n_layers_enc ${N_LAYERS_ENC} --pair_enc ${PAIR_ENC} --n_layers_highway ${n_layers_highway} --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} --bpp_method ${BPP_METHOD} --bpp_base ${BPP_BASE} --optimizer ${OPTIMIZER} --lr ${LR} --min_lr ${min_lr} --lr_decay_factor ${LR_DECAY} --weight_decay ${WEIGHT_DECAY} --val_interval ${VAL_INTERVAL} --max_vals ${MAX_VALS} --task_ordering ${TASK_ORDERING} --scheduler_threshold ${SCHED_THRESH} --load_model ${LOAD_MODEL} --load_tasks ${LOAD_TASKS} --load_preproc ${LOAD_PREPROC} --should_train ${SHOULD_TRAIN} --preproc_file ${PREPROC_FILE}"
eval ${ALLEN_CMD}
