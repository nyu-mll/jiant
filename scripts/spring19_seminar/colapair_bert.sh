# NPI minimal pair experiments on cola-analysis.
# This is part of the spring 19 ling-3340 seminar course.
# Before running the following code, be sure to set up environment,
# get NPI data in place,
# and set project directory to where you want to store the records & saved models

# bert frozen
python main.py --config_file "config/spring19_seminar/colapair_bertF.conf" --override "exp_name = npi_bertraw, run_name = run_bertraw"

# bert tuned on cola
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertnone, run_name = run_bertnone_cola, target_tasks = \"cola,cola-pair-tuned\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = frozen, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_cola/model_state_cola_best.th\", use_classifier=\"cola\"" 

# bert tuned on cola_npi_adv
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertnone, run_name = run_bertnone_cola_npi_adv, target_tasks = \"cola_npi_adv,cola-pair-tuned\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = frozen, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertnone/run_bertnone_cola_npi_adv/model_state_cola_npi_adv_best.th\", use_classifier=\"cola_npi_adv\"

