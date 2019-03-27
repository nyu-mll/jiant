
#!/bin/bash

# Quick-start: run this
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
source activate jiant_new	

eval_cmd="do_target_task_training = 0, do_full_eval = 1, batch_size = 8, write_preds = test, write_strict_glue_format = 1"

## GAP training with RNN sent encoder,  lr = 0.0001 ##
python main.py --config_file config/baselinesuperglue/superglue.conf  --overrides "pretrain_tasks = gap-coreference, do_pretrain = 1,  run_name = rnn_enc, project_dir=gap, bert_embeddings_mode = top, exp_name= base_bert_cased, sent_enc = \"rnn\", bert_model_name = bert-base-cased"
python main.py --config_file config/baselinesuperglue/superglue.conf  --overrides "pretrain_tasks = gap-coreference, target_tasks=gap-coreference, do_target_task_training = 1, do_full_eval = 1, run_name = rnn_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, bert_model_name = bert-base-cased, sent_enc = \"rnn\""
python main.py --config_file config/baselinesuperglue/superglue.conf --overrides "pretrain_tasks = gap-coreference, target_tasks=gap-coreference, run_name = rnn_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, sent_enc = \"rnn\", bert_model_name = bert-base-cased, ${eval_cmd}"

## GAP training with no sent encoder, lr = 0.0001 ##
python main.py --config_file config/baselinesuperglue/superglue.conf  --overrides "pretrain_tasks = gap-coreference, do_pretrain = 1,  run_name = null_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, sent_enc = \"null\", bert_model_name = bert-base-cased"
python main.py --config_file config/baselinesuperglue/superglue.conf  --overrides "pretrain_tasks = gap-coreference, target_tasks=gap-coreference, do_target_task_training = 1,do_full_eval = 1, run_name = null_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, bert_model_name = bert-base-cased, sent_enc = \"null\""
python main.py --config_file config/baselinesuperglue/superglue.conf --overrides "pretrain_tasks = gap-coreference, target_tasks=gap-coreference, run_name = null_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, sent_enc = \"null\", bert_model_name = bert-base-cased, ${eval_cmd}"
