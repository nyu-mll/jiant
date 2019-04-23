# NPI minimal pair experiments on cola-analysis.
# This is part of the spring 19 ling-3340 seminar course.
# Before running the following code, be sure to set up environment,
# get NPI data in place,
# and set project directory to where you want to store the records & saved models

# bert frozen minimal pairs
python main.py --config_file "config/spring19_seminar/colapair_bertF.conf" --override "exp_name = cola-pair, run_name = bert_frozen"

# bert tune minimal pairs
python main.py --config_file "config/spring19_seminar/colapair_bertT.conf" --override "exp_name = cola-pair, run_name = bert_tune"