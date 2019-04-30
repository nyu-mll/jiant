# NPI minimal pair experiments on cola-analysis.
# This is part of the spring 19 ling-3340 seminar course.
# Before running the following code, be sure to set up environment,
# get NPI data in place,
# and set project directory to where you want to store the records & saved models
    

# bertccg tuned on cola
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola, target_tasks = \"cola,cola-pair-tuned\", use_classifier=\"cola\""


# bertccg tuned on cola_npi_sup
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola_npi_sup, target_tasks = \"cola_npi_sup,cola-pair-tuned\", use_classifier=\"cola_npi_sup\""


# bertccg tuned on cola_npi_quessmp
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola_npi_quessmp, target_tasks = \"cola_npi_quessmp,cola-pair-tuned\", use_classifier=\"cola_npi_quessmp\""


# bertccg tuned on cola_npi_ques
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola_npi_ques, target_tasks = \"cola_npi_ques,cola-pair-tuned\", use_classifier=\"cola_npi_ques\""


# bertccg tuned on cola_npi_qnt
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola_npi_qnt, target_tasks = \"cola_npi_qnt,cola-pair-tuned\", use_classifier=\"cola_npi_qnt\""


# bertccg tuned on cola_npi_only
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola_npi_only, target_tasks = \"cola_npi_only,cola-pair-tuned\", use_classifier=\"cola_npi_only\""


# bertccg tuned on cola_npi_negsent
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola_npi_negsent, target_tasks = \"cola_npi_negsent,cola-pair-tuned\", use_classifier=\"cola_npi_negsent\""


# bertccg tuned on cola_npi_negdet
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola_npi_negdet, target_tasks = \"cola_npi_negdet,cola-pair-tuned\", use_classifier=\"cola_npi_negdet\""


# bertccg tuned on cola_npi_cond
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola_npi_cond, target_tasks = \"cola_npi_cond,cola-pair-tuned\", use_classifier=\"cola_npi_cond\""


# bertccg tuned on cola_npi_adv
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola_npi_adv, target_tasks = \"cola_npi_adv,cola-pair-tuned\", use_classifier=\"cola_npi_adv\""


# bertccg tuned on hd_cola_npi_sup
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_cola_npi_sup, target_tasks = \"hd_cola_npi_sup,cola-pair-tuned\", use_classifier=\"hd_cola_npi_sup\""


# bertccg tuned on hd_cola_npi_quessmp
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_cola_npi_quessmp, target_tasks = \"hd_cola_npi_quessmp,cola-pair-tuned\", use_classifier=\"hd_cola_npi_quessmp\""


# bertccg tuned on hd_cola_npi_ques
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_cola_npi_ques, target_tasks = \"hd_cola_npi_ques,cola-pair-tuned\", use_classifier=\"hd_cola_npi_ques\""


# bertccg tuned on hd_cola_npi_qnt
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_cola_npi_qnt, target_tasks = \"hd_cola_npi_qnt,cola-pair-tuned\", use_classifier=\"hd_cola_npi_qnt\""


# bertccg tuned on hd_cola_npi_only
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_cola_npi_only, target_tasks = \"hd_cola_npi_only,cola-pair-tuned\", use_classifier=\"hd_cola_npi_only\""


# bertccg tuned on hd_cola_npi_negsent
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_cola_npi_negsent, target_tasks = \"hd_cola_npi_negsent,cola-pair-tuned\", use_classifier=\"hd_cola_npi_negsent\""


# bertccg tuned on hd_cola_npi_negdet
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_cola_npi_negdet, target_tasks = \"hd_cola_npi_negdet,cola-pair-tuned\", use_classifier=\"hd_cola_npi_negdet\""


# bertccg tuned on hd_cola_npi_cond
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_cola_npi_cond, target_tasks = \"hd_cola_npi_cond,cola-pair-tuned\", use_classifier=\"hd_cola_npi_cond\""


# bertccg tuned on hd_cola_npi_adv
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_cola_npi_adv, target_tasks = \"hd_cola_npi_adv,cola-pair-tuned\", use_classifier=\"hd_cola_npi_adv\""


# bertccg tuned on all_cola_npi
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" --overrides "exp_name = npi_bertccg, run_name = run_bertccg_all_cola_npi, target_tasks = \"all_cola_npi,cola-pair-tuned\", use_classifier=\"all_cola_npi\""
