#!/bin/bash
# 
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mail-type=end
#SBATCH --mail-user=aw3272@nyu.edu

# SBATCH -t 4-00:00
# SBATCH --gres=gpu:p40:1

# This should train an SST model to a validation accuracy of at least 70% in a minute or two.

# cd to jiant root directory
pushd "$(dirname $0)/../"

# Defaults if not already set.
FASTTEXT_EMBS_FILE="${FASTTEXT_EMBS_FILE:-'.'}"
FASTTEXT_MODEL_FILE="${FASTTEXT_MODEL_FILE:-'.'}"
WORD_EMBS_FILE="${WORD_EMBS_FILE:-'.'}"

# machine-specific paths
# Contains JIANT_PROJECT_PREFIX, JIANT_DATA_DIR, WORD_EMBS_FILE and optionally
# FASTTEXT_EMBS_FILE and FASTTEXT_MODEL_FILE
if [ -e user_config.sh ]; then
  echo "Loading environment from ${PWD}/user_config.sh"
  source user_config.sh
fi

EXP_NAME='jiant-demo'
RUN_NAME="sst"
GPUID=0
SEED=42
no_tqdm=0

SHOULD_TRAIN=1
LOAD_MODEL=0
RELOAD_TASKS=0
RELOAD_INDEX=0
RELOAD_VOCAB=0
FORCE_LOAD_EPOCH=-1

train_tasks='sst'
eval_tasks='none'
CLASSIFIER=mlp
d_hid_cls=64
max_seq_len=10
VOCAB_SIZE=30000

word_embs=fastText
fastText=0
char_embs=1
d_word=300
ELMO=0
deep_elmo=0
COVE=0

sent_enc="rnn"
bidirectional=1
d_hid=128
PAIR_ENC="simple"
N_LAYERS_ENC=1
n_layers_highway=0
n_heads=8
d_proj=64
d_ff=2048
warmup=4000

OPTIMIZER="adam"
LR=.001
min_lr=1e-5
dropout=.2
LR_DECAY=.5
patience=50
task_patience=50
WEIGHT_DECAY=0.0
SCHED_THRESH=0.0
BATCH_SIZE=16
BPP_BASE=100
VAL_INTERVAL=100
MAX_VALS=1000
weighting_method="uniform"
scaling_method='none'

source ./src/run_from_vars.sh
