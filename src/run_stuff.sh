#!/bin/bash
# 
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mail-type=end
#SBATCH --mail-user=aw3272@nyu.edu

# SBATCH -t 4-00:00
# SBATCH --gres=gpu:p40:1

SCRATCH_PREFIX='/misc/vlgscratch4/BowmanGroup/awang/'
#SCRATCH_PREFIX='/beegfs/aw3272/'
PROJECT_NAME='mtl-sent-rep'
EXP_NAME="debug"
RUN_NAME="debug"
GPUID=0
SEED=19
no_tqdm=0

SHOULD_TRAIN=1
LOAD_MODEL=0
RELOAD_TASKS=0
RELOAD_INDEX=0
RELOAD_VOCAB=0
load_epoch=-1

train_tasks='all'
eval_tasks='none'
CLASSIFIER=mlp
d_hid_cls=512
max_seq_len=40
VOCAB_SIZE=30000
WORD_EMBS_FILE="${SCRATCH_PREFIX}/raw_data/GloVe/glove.840B.300d.txt"

d_word=300
d_hid=512
glove=1
ELMO=0
deep_elmo=0
elmo_no_glove=0
COVE=0

PAIR_ENC="simple"
N_LAYERS_ENC=1
n_layers_highway=0

OPTIMIZER="sgd"
LR=.1
min_lr=1e-5
dropout=.2
LR_DECAY=.5
patience=5
task_patience=0
train_words=0
WEIGHT_DECAY=0.0
SCHED_THRESH=0.0
BATCH_SIZE=64
BPP_METHOD="percent_tr"
BPP_BASE=10
VAL_INTERVAL=10
MAX_VALS=100
TASK_ORDERING="random"
weighting_method="uniform"
scaling_method='none'

while getopts 'ivkmn:r:S:s:tvh:l:L:o:T:E:O:b:H:p:edcgP:qB:V:M:D:C:X:GI:N:y:K:W:' flag; do
    case "${flag}" in
        P) SCRATCH_PREFIX="${OPTARG}" ;;
        n) EXP_NAME="${OPTARG}" ;;
        r) RUN_NAME="${OPTARG}" ;;
        S) SEED="${OPTARG}" ;;
        q) no_tqdm=1 ;;
        t) SHOULD_TRAIN=0 ;;
        k) LOAD_TASKS=0 ;;
        m) LOAD_MODEL=1 ;;
        i) RELOAD_INDEX=1 ;;
        v) RELOAD_VOCAB=1 ;;
        M) BPP_METHOD="${OPTARG}" ;; 
        B) BPP_BASE="${OPTARG}" ;;
        V) VAL_INTERVAL="${OPTARG}" ;;
        X) MAX_VALS="${OPTARG}" ;;
        T) train_tasks="${OPTARG}" ;;
        #E) eval_tasks="${OPTARG}" ;;
        O) TASK_ORDERING="${OPTARG}" ;;
        H) n_layers_highway="${OPTARG}" ;;
        l) LR="${OPTARG}" ;;
        #s) min_lr="${OPTARG}" ;;
        L) N_LAYERS_ENC="${OPTARG}" ;;
        o) OPTIMIZER="${OPTARG}" ;;
        h) d_hid="${OPTARG}" ;;
        b) BATCH_SIZE="${OPTARG}" ;;
        E) PAIR_ENC="${OPTARG}" ;;
        G) glove=0 ;;
        e) ELMO=1 ;;
        d) deep_elmo=1 ;;
        g) elmo_no_glove=1 ;;
        c) COVE=1 ;;
        D) dropout="${OPTARG}" ;;
        C) CLASSIFIER="${OPTARG}" ;;
        I) GPUID="${OPTARG}" ;;
        N) load_epoch="${OPTARG}" ;;
        y) LR_DECAY="${OPTARG}" ;;
        K) task_patience="${OPTARG}" ;;
        p) patience="${OPTARG}" ;;
        W) weighting_method="${OPTARG}" ;;
        s) scaling_method="${OPTARG}" ;;
    esac
done

LOG_PATH="${SCRATCH_PREFIX}/ckpts/${PROJECT_NAME}/${EXP_NAME}/${RUN_NAME}/log.log"
EXP_DIR="${SCRATCH_PREFIX}/ckpts/${PROJECT_NAME}/${EXP_NAME}/"
RUN_DIR="${SCRATCH_PREFIX}/ckpts/${PROJECT_NAME}/${EXP_NAME}/${RUN_NAME}"
mkdir -p ${EXP_DIR}
mkdir -p ${RUN_DIR}

ALLEN_CMD="python main.py --cuda ${GPUID} --random_seed ${SEED} --no_tqdm ${no_tqdm} --log_file ${LOG_PATH} --exp_dir ${EXP_DIR} --run_dir ${RUN_DIR} --train_tasks ${train_tasks} --eval_tasks ${eval_tasks} --classifier ${CLASSIFIER} --classifier_hid_dim ${d_hid_cls} --max_seq_len ${max_seq_len} --max_word_v_size ${VOCAB_SIZE} --word_embs_file ${WORD_EMBS_FILE} --train_words ${train_words} --glove ${glove} --elmo ${ELMO} --deep_elmo ${deep_elmo} --elmo_no_glove ${elmo_no_glove} --cove ${COVE} --d_word ${d_word} --d_hid ${d_hid} --n_layers_enc ${N_LAYERS_ENC} --pair_enc ${PAIR_ENC} --n_layers_highway ${n_layers_highway} --batch_size ${BATCH_SIZE} --bpp_method ${BPP_METHOD} --bpp_base ${BPP_BASE} --optimizer ${OPTIMIZER} --lr ${LR} --min_lr ${min_lr} --lr_decay_factor ${LR_DECAY} --task_patience ${task_patience} --patience ${patience} --weight_decay ${WEIGHT_DECAY} --dropout ${dropout} --val_interval ${VAL_INTERVAL} --max_vals ${MAX_VALS} --task_ordering ${TASK_ORDERING} --weighting_method ${weighting_method} --scaling_method ${scaling_method} --scheduler_threshold ${SCHED_THRESH} --load_model ${LOAD_MODEL} --reload_tasks ${RELOAD_TASKS} --reload_indexing ${RELOAD_INDEX} --reload_vocab ${RELOAD_VOCAB} --should_train ${SHOULD_TRAIN} --load_epoch ${load_epoch}"
eval ${ALLEN_CMD}
