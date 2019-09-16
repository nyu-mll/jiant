# NPI minimal pair experiments on cola-analysis.
# This is part of the spring 19 ling-3340 seminar course.
# Before running the following code, be sure to set up environment,
# get NPI data in place,
# and set project directory to where you want to store the records & saved models
    

# bertccg tuned on cola
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_cola, target_tasks=\"cola,npi_pair_tuned\", use_classifier=\"cola\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_cola/model_state_cola_best.th\""


# bertccg tuned on npi_sup
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_npi_sup, target_tasks=\"npi_sup,npi_pair_tuned\", use_classifier=\"npi_sup\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_sup/model_state_npi_sup_best.th\""


# bertccg tuned on npi_quessmp
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_npi_quessmp, target_tasks=\"npi_quessmp,npi_pair_tuned\", use_classifier=\"npi_quessmp\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_quessmp/model_state_npi_quessmp_best.th\""


# bertccg tuned on npi_ques
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_npi_ques, target_tasks=\"npi_ques,npi_pair_tuned\", use_classifier=\"npi_ques\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_ques/model_state_npi_ques_best.th\""


# bertccg tuned on npi_qnt
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_npi_qnt, target_tasks=\"npi_qnt,npi_pair_tuned\", use_classifier=\"npi_qnt\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_qnt/model_state_npi_qnt_best.th\""


# bertccg tuned on npi_only
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_npi_only, target_tasks=\"npi_only,npi_pair_tuned\", use_classifier=\"npi_only\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_only/model_state_npi_only_best.th\""


# bertccg tuned on npi_negsent
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_npi_negsent, target_tasks=\"npi_negsent,npi_pair_tuned\", use_classifier=\"npi_negsent\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_negsent/model_state_npi_negsent_best.th\""


# bertccg tuned on npi_negdet
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_npi_negdet, target_tasks=\"npi_negdet,npi_pair_tuned\", use_classifier=\"npi_negdet\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_negdet/model_state_npi_negdet_best.th\""


# bertccg tuned on npi_cond
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_npi_cond, target_tasks=\"npi_cond,npi_pair_tuned\", use_classifier=\"npi_cond\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_cond/model_state_npi_cond_best.th\""


# bertccg tuned on npi_adv
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_npi_adv, target_tasks=\"npi_adv,npi_pair_tuned\", use_classifier=\"npi_adv\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_adv/model_state_npi_adv_best.th\""


# bertccg tuned on hd_npi_sup
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_hd_npi_sup, target_tasks=\"hd_npi_sup,npi_pair_tuned\", use_classifier=\"hd_npi_sup\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_sup/model_state_hd_npi_sup_best.th\""


# bertccg tuned on hd_npi_quessmp
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_hd_npi_quessmp, target_tasks=\"hd_npi_quessmp,npi_pair_tuned\", use_classifier=\"hd_npi_quessmp\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_quessmp/model_state_hd_npi_quessmp_best.th\""


# bertccg tuned on hd_npi_ques
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_hd_npi_ques, target_tasks=\"hd_npi_ques,npi_pair_tuned\", use_classifier=\"hd_npi_ques\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_ques/model_state_hd_npi_ques_best.th\""


# bertccg tuned on hd_npi_qnt
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_hd_npi_qnt, target_tasks=\"hd_npi_qnt,npi_pair_tuned\", use_classifier=\"hd_npi_qnt\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_qnt/model_state_hd_npi_qnt_best.th\""


# bertccg tuned on hd_npi_only
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_hd_npi_only, target_tasks=\"hd_npi_only,npi_pair_tuned\", use_classifier=\"hd_npi_only\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_only/model_state_hd_npi_only_best.th\""


# bertccg tuned on hd_npi_negsent
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_hd_npi_negsent, target_tasks=\"hd_npi_negsent,npi_pair_tuned\", use_classifier=\"hd_npi_negsent\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_negsent/model_state_hd_npi_negsent_best.th\""


# bertccg tuned on hd_npi_negdet
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_hd_npi_negdet, target_tasks=\"hd_npi_negdet,npi_pair_tuned\", use_classifier=\"hd_npi_negdet\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_negdet/model_state_hd_npi_negdet_best.th\""


# bertccg tuned on hd_npi_cond
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_hd_npi_cond, target_tasks=\"hd_npi_cond,npi_pair_tuned\", use_classifier=\"hd_npi_cond\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_cond/model_state_hd_npi_cond_best.th\""


# bertccg tuned on hd_npi_adv
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_hd_npi_adv, target_tasks=\"hd_npi_adv,npi_pair_tuned\", use_classifier=\"hd_npi_adv\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_adv/model_state_hd_npi_adv_best.th\""


# bertccg tuned on all_npi
python main.py --config_file "config/spring19_seminar/npipair_bert.conf" --overrides "exp_name=npi_bertccg, run_name=run_bertccg_all_npi, target_tasks=\"all_npi,npi_pair_tuned\", use_classifier=\"all_npi\", load_target_train_checkpoint=\"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_all_npi/model_state_all_npi_best.th\""
