#!/bin/bash
eval_cmd="do_target_task_training = 0, do_full_eval = 1, batch_size = 8, write_preds = test, write_strict_glue_format = 1"

## GAP training with RNN sent encoder,  lr = 0.0001 ##
python main.py --config_file config/baselinesuperglue/superglue.conf  --overrides "pretrain_tasks = gap-coreference, do_pretrain = 1,  run_name = rnn_enc, project_dir=gap, bert_embeddings_mode = top, exp_name= base_bert_cased, sent_enc = \"rnn\", bert_model_name = bert-base-cased"
python main.py --config_file config/baselinesuperglue/superglue.conf  --overrides "pretrain_tasks = gap-coreference, target_tasks=gap-coreference, do_target_task_training = 1, do_full_eval = 1, run_name = rnn_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, bert_model_name = bert-base-cased, sent_enc = \"rnn\""
python main.py --config_file config/baselinesuperglue/superglue.conf --overrides "pretrain_tasks = gap-coreference, target_tasks=gap-coreference, run_name = rnn_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, sent_enc = \"rnn\", bert_model_name = bert-base-cased, ${eval_cmd}"

## GAP training with no sent encoder, lr = 0.0001 ##
python main.py --config_file config/baselinesuperglue/superglue.conf  --overrides "pretrain_tasks = gap-coreference, do_pretrain = 1,  run_name = null_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, sent_enc = \"null\", bert_model_name = bert-base-cased"
python main.py --config_file config/baselinesuperglue/superglue.conf  --overrides "pretrain_tasks = gap-coreference, target_tasks=gap-coreference, do_target_task_training = 1,do_full_eval = 1, run_name = null_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, bert_model_name = bert-base-cased, sent_enc = \"null\""
python main.py --config_file config/baselinesuperglue/superglue.conf --overrides "pretrain_tasks = gap-coreference, target_tasks=gap-coreference, run_name = null_enc, exp_name= base_bert_cased, project_dir=gap, bert_embeddings_mode = top, sent_enc = \"null\", bert_model_name = bert-base-cased, ${eval_cmd}"
