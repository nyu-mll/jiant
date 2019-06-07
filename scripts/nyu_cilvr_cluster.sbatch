#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.
# If you're using a SLURM-managed cluster, modify this to suit your environment.

#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --constraint=gpu_12gb,pascal

# Example usage:
# JIANT_OVERRIDES="exp_name = main-multi-task-glue, pretrain_tasks = glue, run_name = noelmo-do2-sd1, elmo_chars_only = 1, dropout = 0.2" JIANT_CONF="config/defaults.conf" sbatch ../nyu_cilvr_cluster.sbatch

# Log what we're running and where.
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_PROJECT_PREFIX="coreference_exp"
export JIANT_DATA_DIR="/beegfs/yp913/jiant/data"
export NFS_PROJECT_PREFIX="/nfs/jsalt/exp/nkim"	
export NFS_DATA_DIR="/misc/vlgscratch4/BowmanGroup/yp913/coreference/jiant"
export WORD_EMBS_FILE="/misc/vlgscratch4/BowmanGroup/yp913/jiant/data/glove.840B.300d.txt"
export FASTTEXT_MODEL_FILE=None	
export FASTTEXT_EMBS_FILE=None	
source activate jiant

function only_gap() {
 python main.py --config_file config/baselinesuperglue/final_bert.conf --overrides "classifier_span_pooling=attn, pretrain_tasks = gap-coreference, do_pretrain=1, do_target_task_training=0, tokenizer=bert-large-uncased, target_tasks=gap-coreference, run_name = null_enc_only_gap, project_dir=glue, bert_embeddings_mode = top,  exp_name=bert_large_uncased, batch_size=4, sep_embs_for_skip=1,  do_full_eval=1, sent_enc = \"null\", bert_model_name = bert-large-uncased"
}
