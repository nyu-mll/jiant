#!/bin/bash
# 
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mail-type=end
#SBATCH --mail-user=aw3272@nyu.edu

# SBATCH -t 4-00:00
# SBATCH --gres=gpu:p40:1

SCRATCH_PREFIX='/misc/vlgscratch4/BowmanGroup/awang/'
PROJECT_NAME='mtl-sent-rep'
DATA_DIR="${SCRATCH_PREFIX}/processed_data/mtl-sentence-representations/"
EXP_NAME="debug"
RUN_NAME="debug"
GPUID=0
SEED=19
no_tqdm=0

SHOULD_TRAIN=1
LOAD_MODEL=0
RELOAD_TASKS=1
RELOAD_INDEX=1
RELOAD_VOCAB=1
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

source ../src/run_from_vars.sh
