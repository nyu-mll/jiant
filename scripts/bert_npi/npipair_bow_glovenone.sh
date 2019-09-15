# NPI minimal pair experiments on cola-analysis.
# This is part of the spring 19 ling-3340 seminar course.
# Before running the following code, be sure to set up environment,
# get NPI data in place,
# and set project directory to where you want to store the records & saved models
    

# bow_glovenone tuned on cola
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola, target_tasks = \"cola,npi_pair_tuned\", use_classifier=\"cola\""


# bow_glovenone tuned on cola_npi_sup
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola_npi_sup, target_tasks = \"cola_npi_sup,npi_pair_tuned\", use_classifier=\"cola_npi_sup\""


# bow_glovenone tuned on cola_npi_quessmp
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola_npi_quessmp, target_tasks = \"cola_npi_quessmp,npi_pair_tuned\", use_classifier=\"cola_npi_quessmp\""


# bow_glovenone tuned on cola_npi_ques
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola_npi_ques, target_tasks = \"cola_npi_ques,npi_pair_tuned\", use_classifier=\"cola_npi_ques\""


# bow_glovenone tuned on cola_npi_qnt
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola_npi_qnt, target_tasks = \"cola_npi_qnt,npi_pair_tuned\", use_classifier=\"cola_npi_qnt\""


# bow_glovenone tuned on cola_npi_only
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola_npi_only, target_tasks = \"cola_npi_only,npi_pair_tuned\", use_classifier=\"cola_npi_only\""


# bow_glovenone tuned on cola_npi_negsent
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola_npi_negsent, target_tasks = \"cola_npi_negsent,npi_pair_tuned\", use_classifier=\"cola_npi_negsent\""


# bow_glovenone tuned on cola_npi_negdet
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola_npi_negdet, target_tasks = \"cola_npi_negdet,npi_pair_tuned\", use_classifier=\"cola_npi_negdet\""


# bow_glovenone tuned on cola_npi_cond
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola_npi_cond, target_tasks = \"cola_npi_cond,npi_pair_tuned\", use_classifier=\"cola_npi_cond\""


# bow_glovenone tuned on cola_npi_adv
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_cola_npi_adv, target_tasks = \"cola_npi_adv,npi_pair_tuned\", use_classifier=\"cola_npi_adv\""


# bow_glovenone tuned on hd_cola_npi_sup
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_hd_cola_npi_sup, target_tasks = \"hd_cola_npi_sup,npi_pair_tuned\", use_classifier=\"hd_cola_npi_sup\""


# bow_glovenone tuned on hd_cola_npi_quessmp
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_hd_cola_npi_quessmp, target_tasks = \"hd_cola_npi_quessmp,npi_pair_tuned\", use_classifier=\"hd_cola_npi_quessmp\""


# bow_glovenone tuned on hd_cola_npi_ques
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_hd_cola_npi_ques, target_tasks = \"hd_cola_npi_ques,npi_pair_tuned\", use_classifier=\"hd_cola_npi_ques\""


# bow_glovenone tuned on hd_cola_npi_qnt
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_hd_cola_npi_qnt, target_tasks = \"hd_cola_npi_qnt,npi_pair_tuned\", use_classifier=\"hd_cola_npi_qnt\""


# bow_glovenone tuned on hd_cola_npi_only
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_hd_cola_npi_only, target_tasks = \"hd_cola_npi_only,npi_pair_tuned\", use_classifier=\"hd_cola_npi_only\""


# bow_glovenone tuned on hd_cola_npi_negsent
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_hd_cola_npi_negsent, target_tasks = \"hd_cola_npi_negsent,npi_pair_tuned\", use_classifier=\"hd_cola_npi_negsent\""


# bow_glovenone tuned on hd_cola_npi_negdet
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_hd_cola_npi_negdet, target_tasks = \"hd_cola_npi_negdet,npi_pair_tuned\", use_classifier=\"hd_cola_npi_negdet\""


# bow_glovenone tuned on hd_cola_npi_cond
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_hd_cola_npi_cond, target_tasks = \"hd_cola_npi_cond,npi_pair_tuned\", use_classifier=\"hd_cola_npi_cond\""


# bow_glovenone tuned on hd_cola_npi_adv
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_hd_cola_npi_adv, target_tasks = \"hd_cola_npi_adv,npi_pair_tuned\", use_classifier=\"hd_cola_npi_adv\""


# bow_glovenone tuned on all_cola_npi
python main.py --config_file "jiant/config/bert_npi/npipair_bow_glove.conf" --overrides "exp_name = npi_bow_glovenone, run_name = run_bow_glovenone_all_cola_npi, target_tasks = \"all_cola_npi,npi_pair_tuned\", use_classifier=\"all_cola_npi\""
