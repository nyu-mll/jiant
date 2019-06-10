# NPI minimal pair experiments on cola-analysis.
# This is part of the spring 19 ling-3340 seminar course.
# Before running the following code, be sure to set up environment,
# get NPI data in place,
# and set project directory to where you want to store the records & saved models
    

# bertmnli tuned on cola
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola, target_tasks = \"cola,npi_pair_tuned\", use_classifier=\"cola\""


# bertmnli tuned on cola_npi_sup
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_sup, target_tasks = \"cola_npi_sup,npi_pair_tuned\", use_classifier=\"cola_npi_sup\""


# bertmnli tuned on cola_npi_quessmp
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_quessmp, target_tasks = \"cola_npi_quessmp,npi_pair_tuned\", use_classifier=\"cola_npi_quessmp\""


# bertmnli tuned on cola_npi_ques
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_ques, target_tasks = \"cola_npi_ques,npi_pair_tuned\", use_classifier=\"cola_npi_ques\""


# bertmnli tuned on cola_npi_qnt
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_qnt, target_tasks = \"cola_npi_qnt,npi_pair_tuned\", use_classifier=\"cola_npi_qnt\""


# bertmnli tuned on cola_npi_only
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_only, target_tasks = \"cola_npi_only,npi_pair_tuned\", use_classifier=\"cola_npi_only\""


# bertmnli tuned on cola_npi_negsent
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_negsent, target_tasks = \"cola_npi_negsent,npi_pair_tuned\", use_classifier=\"cola_npi_negsent\""


# bertmnli tuned on cola_npi_negdet
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_negdet, target_tasks = \"cola_npi_negdet,npi_pair_tuned\", use_classifier=\"cola_npi_negdet\""


# bertmnli tuned on cola_npi_cond
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_cond, target_tasks = \"cola_npi_cond,npi_pair_tuned\", use_classifier=\"cola_npi_cond\""


# bertmnli tuned on cola_npi_adv
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_adv, target_tasks = \"cola_npi_adv,npi_pair_tuned\", use_classifier=\"cola_npi_adv\""


# bertmnli tuned on hd_cola_npi_sup
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_sup, target_tasks = \"hd_cola_npi_sup,npi_pair_tuned\", use_classifier=\"hd_cola_npi_sup\""


# bertmnli tuned on hd_cola_npi_quessmp
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_quessmp, target_tasks = \"hd_cola_npi_quessmp,npi_pair_tuned\", use_classifier=\"hd_cola_npi_quessmp\""


# bertmnli tuned on hd_cola_npi_ques
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_ques, target_tasks = \"hd_cola_npi_ques,npi_pair_tuned\", use_classifier=\"hd_cola_npi_ques\""


# bertmnli tuned on hd_cola_npi_qnt
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_qnt, target_tasks = \"hd_cola_npi_qnt,npi_pair_tuned\", use_classifier=\"hd_cola_npi_qnt\""


# bertmnli tuned on hd_cola_npi_only
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_only, target_tasks = \"hd_cola_npi_only,npi_pair_tuned\", use_classifier=\"hd_cola_npi_only\""


# bertmnli tuned on hd_cola_npi_negsent
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_negsent, target_tasks = \"hd_cola_npi_negsent,npi_pair_tuned\", use_classifier=\"hd_cola_npi_negsent\""


# bertmnli tuned on hd_cola_npi_negdet
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_negdet, target_tasks = \"hd_cola_npi_negdet,npi_pair_tuned\", use_classifier=\"hd_cola_npi_negdet\""


# bertmnli tuned on hd_cola_npi_cond
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_cond, target_tasks = \"hd_cola_npi_cond,npi_pair_tuned\", use_classifier=\"hd_cola_npi_cond\""


# bertmnli tuned on hd_cola_npi_adv
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_adv, target_tasks = \"hd_cola_npi_adv,npi_pair_tuned\", use_classifier=\"hd_cola_npi_adv\""


# bertmnli tuned on all_cola_npi
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_all_cola_npi, target_tasks = \"all_cola_npi,npi_pair_tuned\", use_classifier=\"all_cola_npi\""
