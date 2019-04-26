#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:1
#SBATCH --job-name=myrun
#SBATCH --output=slurm_%j.out

# This script loads 48 pretrained checkpoints and run corresponding probing tasks
# 4 models: bert, bertccg, bertmnli, bow) 
# 12 pre-training settings: plain, cola, npiALL, npiNOadv, npiNOcond, npiNOnegdet, npiNOnegsent, npiNOonly, npiNOqnt, npiNOques, npiNPquessmp, npiNOsup
# bert_plain and bow_plain require no pre-training, so no loading is needed
# 3 experiment names, with run names being all 48 combinations (saves time; tasks only needed to be created once per all 48 runs)
# having bow_plain with bow_ runs in the same folder seems to create issues

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert_plain, run_name = bert_plain, load_eval_checkpoint = none, allow_untrained_encoder_parameters = 1"

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_cola, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiALL, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiNOadv, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_adv_li,npi_adv_sc,npi_adv_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiNOcond, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_cond_li,npi_cond_sc,npi_cond_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiNOnegdet, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_negdet_li,npi_negsent_sc,npi_negsent_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiNOnegsent, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_negsent_li,npi_negsent_sc,npi_negsent_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiNOonly, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_only_li,npi_only_sc,npi_only_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiNOqnt, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_qnt_li,npi_qnt_sc,npi_qnt_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiNOques, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_ques_li,npi_ques_sc,npi_ques_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiNOquessmp, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_quessmp_li,npi_quessmp_sc,npi_quessmp_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_npiNOsup, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_sup_li,npi_sup_sc,npi_sup_pr\""


python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_plain, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_cola, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiALL, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiNOadv, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_adv_li,npi_adv_sc,npi_adv_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiNOcond, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_cond_li,npi_cond_sc,npi_cond_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiNOnegdet, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_negdet_li,npi_negsent_sc,npi_negsent_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiNOnegsent, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_negsent_li,npi_negsent_sc,npi_negsent_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiNOonly, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_only_li,npi_only_sc,npi_only_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiNOqnt, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_qnt_li,npi_qnt_sc,npi_qnt_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiNOques, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_ques_li,npi_ques_sc,npi_ques_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiNOquessmp, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_quessmp_li,npi_quessmp_sc,npi_quessmp_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_npiNOsup, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_sup_li,npi_sup_sc,npi_sup_pr\""


python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_plain, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_cola, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiALL, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiNOadv, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_adv_li,npi_adv_sc,npi_adv_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiNOcond, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_cond_li,npi_cond_sc,npi_cond_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiNOnegdet, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_negdet_li,npi_negsent_sc,npi_negsent_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiNOnegsent, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_negsent_li,npi_negsent_sc,npi_negsent_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiNOonly, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_only_li,npi_only_sc,npi_only_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiNOqnt, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_qnt_li,npi_qnt_sc,npi_qnt_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiNOques, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_ques_li,npi_ques_sc,npi_ques_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiNOquessmp, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_quessmp_li,npi_quessmp_sc,npi_quessmp_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_npiNOsup, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_sup_li,npi_sup_sc,npi_sup_pr\""


python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_plain, run_name = bow_plain, load_eval_checkpoint = none, sent_enc = \"bow\", word_embs = \"glove\", tokenizer = \"MosesTokenizer\", bert_model_name = \"\", skip_embs = 0, allow_untrained_encoder_parameters = 1, allow_missing_task_map = 1"

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_cola, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiALL, load_eval_checkpoint = \"PATH_OF_MODEL\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiNOadv, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_adv_li,npi_adv_sc,npi_adv_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiNOcond, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_cond_li,npi_cond_sc,npi_cond_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiNOnegdet, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_negdet_li,npi_negsent_sc,npi_negsent_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiNOnegsent, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_negsent_li,npi_negsent_sc,npi_negsent_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiNOonly, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_only_li,npi_only_sc,npi_only_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiNOqnt, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_qnt_li,npi_qnt_sc,npi_qnt_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiNOques, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_ques_li,npi_ques_sc,npi_ques_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiNOquessmp, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_quessmp_li,npi_quessmp_sc,npi_quessmp_pr\""

python main.py --config_file config/spring19_seminar/cola_var/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow, run_name = bow_npiNOsup, load_eval_checkpoint = \"PATH_OF_MODEL\", target_tasks = \"npi_sup_li,npi_sup_sc,npi_sup_pr\""

