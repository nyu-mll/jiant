#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --job-name=myrun
#SBATCH --output=slurm_%j.out

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola, target_tasks = "cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_adv, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", pretrain_tasks = "ccg,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_cond, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_cond", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_negdet, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_negdet", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_negsent, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_negsent", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_only, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_only", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_qnt, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_qnt", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_ques, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_ques", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_quessmp, target_tasks = "cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_quessmp", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_sup, target_tasks = "cola_npi_sup", pretrain_tasks = "ccg,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_sup, target_tasks = "cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_sup", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_quessmp, target_tasks = "cola_npi_quessmp", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_ques, target_tasks = "cola_npi_ques", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_qnt, target_tasks = "cola_npi_qnt", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_only, target_tasks = "cola_npi_only", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_negsent, target_tasks = "cola_npi_negsent", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_negdet, target_tasks = "cola_npi_negdet", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_cond, target_tasks = "cola_npi_cond", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_adv, target_tasks = "cola_npi_adv", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_allnpi, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola, target_tasks = "cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_adv, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", pretrain_tasks = "mnli,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_cond, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_cond", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_negdet, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_negdet", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_negsent, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_negsent", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_only, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_only", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_qnt, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_qnt", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_ques, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_ques", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_quessmp, target_tasks = "cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_quessmp", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_sup, target_tasks = "cola_npi_sup", pretrain_tasks = "mnli,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_sup, target_tasks = "cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_sup", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_quessmp, target_tasks = "cola_npi_quessmp", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_ques, target_tasks = "cola_npi_ques", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_qnt, target_tasks = "cola_npi_qnt", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_only, target_tasks = "cola_npi_only", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_negsent, target_tasks = "cola_npi_negsent", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_negdet, target_tasks = "cola_npi_negdet", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_cond, target_tasks = "cola_npi_cond", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_adv, target_tasks = "cola_npi_adv", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_allnpi, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola, target_tasks = "cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola_npi_adv, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", pretrain_tasks = "none,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola_npi_cond, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", pretrain_tasks = "none,cola_npi_cond", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola_npi_negdet, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_negdet", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola_npi_negsent, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_negsent", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola_npi_only, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_only", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola_npi_qnt, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_qnt", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola_npi_ques, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_ques", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola_npi_quessmp, target_tasks = "cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_quessmp", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_hd_cola_npi_sup, target_tasks = "cola_npi_sup", pretrain_tasks = "none,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_cola_npi_sup, target_tasks = "cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_sup", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_hd_cola_npi_quessmp, target_tasks = "cola_npi_quessmp", pretrain_tasks = "none,cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_hd_cola_npi_ques, target_tasks = "cola_npi_ques", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_hd_cola_npi_qnt, target_tasks = "cola_npi_qnt", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_hd_cola_npi_only, target_tasks = "cola_npi_only", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_hd_cola_npi_negsent, target_tasks = "cola_npi_negsent", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_hd_cola_npi_negdet, target_tasks = "cola_npi_negdet", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_hd_cola_npi_cond, target_tasks = "cola_npi_cond", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_hd_cola_npi_adv, target_tasks = "cola_npi_adv", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertnone_allnpi, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola, target_tasks = "cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_adv, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", pretrain_tasks = "ccg,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_cond, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_cond", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_negdet, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_negdet", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_negsent, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_negsent", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_only, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_only", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_qnt, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_qnt", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_ques, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_ques", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_quessmp, target_tasks = "cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_quessmp", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_sup, target_tasks = "cola_npi_sup", pretrain_tasks = "ccg,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_sup, target_tasks = "cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_sup", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_quessmp, target_tasks = "cola_npi_quessmp", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_ques, target_tasks = "cola_npi_ques", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_qnt, target_tasks = "cola_npi_qnt", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_only, target_tasks = "cola_npi_only", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_negsent, target_tasks = "cola_npi_negsent", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_negdet, target_tasks = "cola_npi_negdet", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_cond, target_tasks = "cola_npi_cond", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_adv, target_tasks = "cola_npi_adv", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_allnpi, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "ccg,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola, target_tasks = "cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_adv, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", pretrain_tasks = "mnli,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_cond, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_cond", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_negdet, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_negdet", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_negsent, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_negsent", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_only, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_only", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_qnt, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_qnt", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_ques, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_ques", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_quessmp, target_tasks = "cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_quessmp", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_sup, target_tasks = "cola_npi_sup", pretrain_tasks = "mnli,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_sup, target_tasks = "cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_sup", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_quessmp, target_tasks = "cola_npi_quessmp", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_ques, target_tasks = "cola_npi_ques", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_qnt, target_tasks = "cola_npi_qnt", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_only, target_tasks = "cola_npi_only", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_negsent, target_tasks = "cola_npi_negsent", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_negdet, target_tasks = "cola_npi_negdet", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_cond, target_tasks = "cola_npi_cond", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_adv, target_tasks = "cola_npi_adv", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_allnpi, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "mnli,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola, target_tasks = "cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola_npi_adv, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", pretrain_tasks = "none,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola_npi_cond, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", pretrain_tasks = "none,cola_npi_cond", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola_npi_negdet, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_negdet", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola_npi_negsent, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_negsent", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola_npi_only, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_only", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola_npi_qnt, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_qnt", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola_npi_ques, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_ques", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola_npi_quessmp, target_tasks = "cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_quessmp", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_hd_cola_npi_sup, target_tasks = "cola_npi_sup", pretrain_tasks = "none,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_cola_npi_sup, target_tasks = "cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_sup", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_hd_cola_npi_quessmp, target_tasks = "cola_npi_quessmp", pretrain_tasks = "none,cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_hd_cola_npi_ques, target_tasks = "cola_npi_ques", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_hd_cola_npi_qnt, target_tasks = "cola_npi_qnt", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_hd_cola_npi_only, target_tasks = "cola_npi_only", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_hd_cola_npi_negsent, target_tasks = "cola_npi_negsent", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_hd_cola_npi_negdet, target_tasks = "cola_npi_negdet", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_hd_cola_npi_cond, target_tasks = "cola_npi_cond", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_hd_cola_npi_adv, target_tasks = "cola_npi_adv", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovenone_allnpi, target_tasks = "cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", pretrain_tasks = "none,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv", do_pretrain = 0, allow_untrained_encoder_parameters = 1" 

