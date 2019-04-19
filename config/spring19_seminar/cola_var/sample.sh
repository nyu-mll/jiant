#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --job-name=myrun
#SBATCH --output=slurm_%j.out

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_acpt, run_name = run_bert_ccg_cola, target_tasks = "cola", pretrain_tasks = "ccg", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_acpt, run_name = run_bert_mnli_cola, target_tasks = "cola", pretrain_tasks = "mnli", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_acpt, run_name = run_bert_none_cola, target_tasks = "cola", pretrain_tasks = "none", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bilstm.conf \
    --overrides "exp_name = npi_acpt, run_name = run_bilstm_ccg_cola, target_tasks = "cola", pretrain_tasks = "ccg", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bilstm.conf \
    --overrides "exp_name = npi_acpt, run_name = run_bilstm_mnli_cola, target_tasks = "cola", pretrain_tasks = "mnli", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bilstm.conf \
    --overrides "exp_name = npi_acpt, run_name = run_bilstm_none_cola, target_tasks = "cola", pretrain_tasks = "none", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_acpt, run_name = run_bow_glove_ccg_cola, target_tasks = "cola", pretrain_tasks = "ccg", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_acpt, run_name = run_bow_glove_mnli_cola, target_tasks = "cola", pretrain_tasks = "mnli", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_acpt, run_name = run_bow_glove_none_cola, target_tasks = "cola", pretrain_tasks = "none", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

