#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:1
#SBATCH --job-name=myrun
#SBATCH --output=slurm_%j.out

# This script loads 46 pretrained checkpoints and runs corresponding probing tasks
# It also runs probing tasks BERT and BOW plain models
# 4 models: bert, bertccg, bertmnli, BOW
# 12 pre-training settings: plain, cola, all_cola_npi, hd_cola_npi_adv, hd_cola_npi_cond, hd_cola_npi_negdet, hd_cola_npi_negsent, hd_cola_npi_only, hd_cola_npi_qnt, hd_cola_npi_ques, hd_cola_npi_quessmp, hd_cola_npi_sup
# bert_plain and bow_plain require no pre-training, so no loading is needed
# 5 experiment names, with run names being all 48 combinations (saves time; tasks only needed to be created three times in total)
# having bow_plain with bow runs in the same folder seems to create issues

#load plain bert, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_plain, load_eval_checkpoint = none, allow_untrained_encoder_parameters = 1"

#load bert+cola, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_cola, load_eval_checkpoint =  \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_cola/model_state_cola_best.th\""

#load bert+all npi, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_all_cola_npi, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_all_cola_npi/model_state_all_cola_npi_best.th\""

#load bert+all npi with adv being held out, train and eval on adv probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_hd_cola_npi_adv, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_hd_cola_npi_adv/model_state_hd_cola_npi_adv_best.th\", target_tasks = \"npi_adv_li,npi_adv_sc,npi_adv_pr\""

#load bert+all npi with cond being held out, train and eval on cond probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_hd_cola_npi_cond, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_hd_cola_npi_cond/model_state_hd_cola_npi_cond_best.th\", target_tasks = \"npi_cond_li,npi_cond_sc,npi_cond_pr\""

#load bert+all npi with negdet being held out, train and eval on negdet probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_hd_cola_npi_negdet, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_hd_cola_npi_negdet/model_state_hd_cola_npi_negdet_best.th\", target_tasks = \"npi_negdet_li,npi_negdet_sc,npi_negdet_pr\""

#load bert+all npi with negsent being held out, train and eval on negsent probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_hd_cola_npi_negsent, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_hd_cola_npi_negsent/model_state_hd_cola_npi_negsent_best.th\", target_tasks = \"npi_negsent_li,npi_negsent_sc,npi_negsent_pr\""

#load bert+all npi with only being held out, train and eval on only probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_hd_cola_npi_only, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_hd_cola_npi_only/model_state_hd_cola_npi_only_best.th\", target_tasks = \"npi_only_li,npi_only_sc,npi_only_pr\""

#load bert+all npi with qnt being held out, train and eval on qnt probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_hd_cola_npi_qnt, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_hd_cola_npi_qnt/model_state_hd_cola_npi_qnt_best.th\", target_tasks = \"npi_qnt_li,npi_qnt_sc,npi_qnt_pr\""

#load bert+all npi with ques being held out, train and eval on ques probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_hd_cola_npi_ques, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_hd_cola_npi_ques/model_state_hd_cola_npi_ques_best.th\", target_tasks = \"npi_ques_li,npi_ques_sc,npi_ques_pr\""

#load bert+all npi with quessmp being held out, train and eval on quessmp probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_hd_cola_npi_quessmp, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_hd_cola_npi_quessmp/model_state_hd_cola_npi_quessmp_best.th\", target_tasks = \"npi_quessmp_li,npi_quessmp_sc,npi_quessmp_pr\""

#load bert+all npi with sup being held out, train and eval on sup probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bert, run_name = bert_hd_cola_npi_sup, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_hd_cola_npi_sup/model_state_hd_cola_npi_sup_best.th\", target_tasks = \"npi_sup_li,npi_sup_sc,npi_sup_pr\""

#load plain bow_glove, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove_plain, run_name = bow_glove_plain, load_eval_checkpoint = none, allow_untrained_encoder_parameters = 1, sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+cola, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_cola, load_eval_checkpoint =  \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_cola/model_state_cola_best.th\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_all_cola_npi, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_all_cola_npi/model_state_all_cola_npi_best.th\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi with adv being held out, train and eval on adv probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_hd_cola_npi_adv, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_hd_cola_npi_adv/model_state_hd_cola_npi_adv_best.th\", target_tasks = \"npi_adv_li,npi_adv_sc,npi_adv_pr\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi with cond being held out, train and eval on cond probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_hd_cola_npi_cond, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_hd_cola_npi_cond/model_state_hd_cola_npi_cond_best.th\", target_tasks = \"npi_cond_li,npi_cond_sc,npi_cond_pr\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi with negdet being held out, train and eval on negdet probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_hd_cola_npi_negdet, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_hd_cola_npi_negdet/model_state_hd_cola_npi_negdet_best.th\", target_tasks = \"npi_negdet_li,npi_negdet_sc,npi_negdet_pr\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi with negsent being held out, train and eval on negsent probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_hd_cola_npi_negsent, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_hd_cola_npi_negsent/model_state_hd_cola_npi_negsent_best.th\", target_tasks = \"npi_negsent_li,npi_negsent_sc,npi_negsent_pr\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi with only being held out, train and eval on only probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_hd_cola_npi_only, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_hd_cola_npi_only/model_state_hd_cola_npi_only_best.th\", target_tasks = \"npi_only_li,npi_only_sc,npi_only_pr\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi with qnt being held out, train and eval on qnt probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_hd_cola_npi_qnt, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_hd_cola_npi_qnt/model_state_hd_cola_npi_qnt_best.th\", target_tasks = \"npi_qnt_li,npi_qnt_sc,npi_qnt_pr\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi with ques being held out, train and eval on ques probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_hd_cola_npi_ques, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_hd_cola_npi_ques/model_state_hd_cola_npi_ques_best.th\", target_tasks = \"npi_ques_li,npi_ques_sc,npi_ques_pr\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi with quessmp being held out, train and eval on quessmp probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_hd_cola_npi_quessmp, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_hd_cola_npi_quessmp/model_state_hd_cola_npi_quessmp_best.th\", target_tasks = \"npi_quessmp_li,npi_quessmp_sc,npi_quessmp_pr\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""

#load bow_glove+all npi with sup being held out, train and eval on sup probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bow_glove, run_name = bow_glove_hd_cola_npi_sup, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bow_glovenone/run_bow_glovenone_hd_cola_npi_sup/model_state_hd_cola_npi_sup_best.th\", target_tasks = \"npi_sup_li,npi_sup_sc,npi_sup_pr\", sent_enc = \"bow\", allow_missing_task_map = 1, bert_model_name = \"\", word_embs = \"glove\", skip_embs = 0, tokenizer = \"MosesTokenizer\""


#load bertcgg plain, train and eval on all probing tasks (MAY HAVE TO SUPPLY MODEL NAME MANUALLY)
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_plain, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_main_epoch_??.best_macro.th\""

#load bertcgg+cola, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_cola, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_cola/model_state_cola_best.th\""

#load bertcgg+all npi, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_all_cola_npi, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_all_cola_npi/model_state_all_cola_npi_best.th\""

#load bertccg+all npi with adv being held out, train and eval on adv probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_hd_cola_npi_adv, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_cola_npi_adv/model_state_hd_cola_npi_adv_best.th\", target_tasks = \"npi_adv_li,npi_adv_sc,npi_adv_pr\""

#load bertccg+all npi with cond being held out, train and eval on cond probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_hd_cola_npi_cond, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_cola_npi_cond/model_state_hd_cola_npi_cond_best.th\", target_tasks = \"npi_cond_li,npi_cond_sc,npi_cond_pr\""

#load bertccg+all npi with negdet being held out, train and eval on negdet probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_hd_cola_npi_negdet, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_cola_npi_negdet/model_state_hd_cola_npi_negdet_best.th\", target_tasks = \"npi_negdet_li,npi_negsent_sc,npi_negsent_pr\""

#load bertccg+all npi with negsent being held out, train and eval on negsent probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_hd_cola_npi_negsent, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_cola_npi_negsent/model_state_hd_cola_npi_negsent_best.th\", target_tasks = \"npi_negsent_li,npi_negsent_sc,npi_negsent_pr\""

#load bertccg+all npi with only being held out, train and eval on only probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_hd_cola_npi_only, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_cola_npi_only/model_state_hd_cola_npi_only_best.th\", target_tasks = \"npi_only_li,npi_only_sc,npi_only_pr\""

#load bertccg+all npi with qnt being held out, train and eval on qnt probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_hd_cola_npi_qnt, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_cola_npi_qnt/model_state_hd_cola_npi_qnt_best.th\", target_tasks = \"npi_qnt_li,npi_qnt_sc,npi_qnt_pr\""

#load bertccg+all npi with ques being held out, train and eval on ques probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_hd_cola_npi_ques, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_cola_npi_ques/model_state_hd_cola_npi_ques_best.th\", target_tasks = \"npi_ques_li,npi_ques_sc,npi_ques_pr\""

#load bertccg+all npi with quessmp being held out, train and eval on quessmp probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_hd_cola_npi_quessmp, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_cola_npi_quessmp/model_state_hd_cola_npi_quessmp_best.th\", target_tasks = \"npi_quessmp_li,npi_quessmp_sc,npi_quessmp_pr\""

#load bertccg+all npi with quessmp being held out, train and eval on sup probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertccg, run_name = bertccg_hd_cola_npi_sup, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_cola_npi_sup/model_state_hd_cola_npi_sup_best.th\", target_tasks = \"npi_sup_li,npi_sup_sc,npi_sup_pr\""


#load bertmnli plain, train and eval on all probing tasks (MAY HAVE TO SUPPLY MODEL NAME MANUALLY)
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_plain, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_main_epoch_??.best_macro.th\""

#load bertmnli+cola, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_cola, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola/model_state_cola_best.th\""

#load bertmnli+all npi, train and eval on all probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_all_cola_npi, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_all_cola_npi/model_state_all_cola_npi_best.th\""

#load bertmnli+all npi with adv being held out, train and eval on adv probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_adv, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_adv/model_state_hd_cola_npi_adv_best.th\", target_tasks = \"npi_adv_li,npi_adv_sc,npi_adv_pr\""

#load bertmnli+all npi with cond being held out, train and eval on cond probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_cond, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_cond/model_state_hd_cola_npi_cond_best.th\", target_tasks = \"npi_cond_li,npi_cond_sc,npi_cond_pr\""

#load bertmnli+all npi with negdet being held out, train and eval on negdet probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_negdet, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_negdet/model_state_hd_cola_npi_negdet_best.th\", target_tasks = \"npi_negdet_li,npi_negsent_sc,npi_negsent_pr\""

#load bertmnli+all npi with negsent being held out, train and eval on negsent probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_negsent, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_negsent/model_state_hd_cola_npi_negsent_best.th\", target_tasks = \"npi_negsent_li,npi_negsent_sc,npi_negsent_pr\""

#load bertmnli+all npi with only being held out, train and eval on only probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_only, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_only/model_state_hd_cola_npi_only_best.th\", target_tasks = \"npi_only_li,npi_only_sc,npi_only_pr\""

#load bertmnli+all npi with qnt being held out, train and eval on qnt probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_qnt, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_qnt/model_state_hd_cola_npi_qnt_best.th\", target_tasks = \"npi_qnt_li,npi_qnt_sc,npi_qnt_pr\""

#load bertmnli+all npi with ques being held out, train and eval on ques probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_ques, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_ques/model_state_hd_cola_npi_ques_best.th\", target_tasks = \"npi_ques_li,npi_ques_sc,npi_ques_pr\""

#load bertmnli+all npi with quessmp being held out, train and eval on quessmp probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_quessmp, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_quessmp/model_state_hd_cola_npi_quessmp_best.th\", target_tasks = \"npi_quessmp_li,npi_quessmp_sc,npi_quessmp_pr\""

#load bertmnli+all npi with sup being held out, train and eval on sup probing tasks
python main.py --config_file config/spring19_seminar/npi_probing_tasks.conf --overrides "exp_name=NPI_probing_bertmnli, run_name = bertmnli_hd_cola_npi_sup, load_eval_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_sup/model_state_hd_cola_npi_sup_best.th\", target_tasks = \"npi_sup_li,npi_sup_sc,npi_sup_pr\""
