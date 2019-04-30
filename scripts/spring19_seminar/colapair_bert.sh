# NPI minimal pair experiments on cola-analysis.
# This is part of the spring 19 ling-3340 seminar course.
# Before running the following code, be sure to set up environment,
# get NPI data in place,
# and set project directory to where you want to store the records & saved models

# bert frozen (masked language modeling)
python main.py --config_file "config/spring19_seminar/colapair_bert_frozen.conf" --override "exp_name = npi_bertmlm, run_name = run_bertmlm"

# bert tuned on cola
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" \
    --overrides "exp_name = npi_bertnone, run_name = run_bertnone_cola, target_tasks = \"cola,cola-pair-tuned\", use_classifier=\"cola\"" 

# bert tuned on cola_npi_adv
python main.py --config_file "config/spring19_seminar/colapair_bert_tuned.conf" \
    --overrides "exp_name = npi_bertnone, run_name = run_bertnone_cola_npi_adv, target_tasks = \"cola_npi_adv,cola-pair-tuned\", use_classifier=\"cola_npi_adv\""

