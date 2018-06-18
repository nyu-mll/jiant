#!/bin/bash
# 
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mail-type=end
#SBATCH --mail-user=aw3272@nyu.edu

# SBATCH -t 4-00:00
# SBATCH --gres=gpu:p40:1

# This should train an SST model to a validation accuracy of at least 70% in a minute or two.

# SET THESE BEFORE RUNNING:
SCRATCH_PREFIX='/Users/Bowman/Drive/JSALT/demo'
DATA_DIR="/Users/Bowman/Drive/JSALT/jiant/glue_data/"

PROJECT_NAME='jiant-demo'
EXP_NAME="cola"
RUN_NAME="cola_1"
GPUID=-1
SEED=42
no_tqdm=0

SHOULD_TRAIN=1
LOAD_MODEL=0
RELOAD_TASKS=1
RELOAD_INDEX=1
RELOAD_VOCAB=1
load_epoch=-1

train_tasks='mnli-fiction'
eval_tasks='mnli'
CLASSIFIER=mlp
d_hid_cls=16
max_seq_len=10
VOCAB_SIZE=1000
WORD_EMBS_FILE="~/glove.840B.300d.txt"

d_word=300
d_hid=32
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
patience=50
task_patience=50
train_words=0
WEIGHT_DECAY=0.0
SCHED_THRESH=0.0
BATCH_SIZE=16
BPP_METHOD="percent_tr"
BPP_BASE=100
VAL_INTERVAL=100
MAX_VALS=1000
TASK_ORDERING="random"
weighting_method="uniform"
scaling_method='none'

source run_from_vars.sh
