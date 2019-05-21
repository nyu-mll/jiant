#!/bin/bash
# Functions to run SuperGLUE BERT baselines.
# Usage: ./scripts/superglue-baselines.sh ${TASK} ${GPU_ID} ${SEED}
#   - TASK: one of {"cb", "copa", "multirc", "rte", "wic", "wsc"}
#   - GPU_ID: GPU to use, or -1 for CPU. Defaults to 0.
#   - SEED: random seed. Defaults to 111.

export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_DATA_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant/data"
export NFS_PROJECT_PREFIX="/nfs/jsalt/exp/nkim" 
export NFS_DATA_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant"
export WORD_EMBS_FILE="/misc/vlgscratch4/BowmanGroup/yp913/jiant/data/glove.840B.300d.txt"
export FASTTEXT_MODEL_FILE=None 
export FASTTEXT_EMBS_FILE=None  
module load anaconda3   
module load cuda 10.0   
source activate jiant_neW


source user_config.sh
seed=${3:-111}
gpuid=${2:-0}


function wsc() {
    # NOTE: We use Adam b/c we were getting weird degenerate runs with BERT Adam
    python main.py --config config/superglue-bert.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = wsc, pretrain_tasks = \"winograd-coreference\", target_tasks = \"winograd-coreference\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 139, optimizer = adam"
}

wsc
