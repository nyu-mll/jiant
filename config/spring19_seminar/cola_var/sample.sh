#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --job-name=myrun
#SBATCH --output=slurm_%j.out

python main.py --config_file config/spring19_pretrain/cola_bert.conf \
    --overrides "exp_name = exp_pretrain, run_name = run_bert_ccg, target_tasks = "cola", pretrain_tasks = "ccg", do_pretrain = 1" 

python main.py --config_file config/spring19_pretrain/cola_bert.conf \
    --overrides "exp_name = exp_pretrain, run_name = run_bert_mnli, target_tasks = "cola", pretrain_tasks = "mnli", do_pretrain = 1" 

python main.py --config_file config/spring19_pretrain/cola_bert.conf \
    --overrides "exp_name = exp_pretrain, run_name = run_bert_none, target_tasks = "cola", pretrain_tasks = "none", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_pretrain/cola_bilstm.conf \
    --overrides "exp_name = exp_pretrain, run_name = run_bilstm_ccg, target_tasks = "cola", pretrain_tasks = "ccg", do_pretrain = 1" 

python main.py --config_file config/spring19_pretrain/cola_bilstm.conf \
    --overrides "exp_name = exp_pretrain, run_name = run_bilstm_mnli, target_tasks = "cola", pretrain_tasks = "mnli", do_pretrain = 1" 

python main.py --config_file config/spring19_pretrain/cola_bilstm.conf \
    --overrides "exp_name = exp_pretrain, run_name = run_bilstm_none, target_tasks = "cola", pretrain_tasks = "none", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 
