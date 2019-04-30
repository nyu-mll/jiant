# NPI minimal pair experiments on cola-analysis.
# This is part of the spring 19 ling-3340 seminar course.
# Before running the following code, be sure to set up environment,
# get NPI data in place,
# and set project directory to where you want to store the records & saved models


# bert frozen (masked language modeling)
python main.py --config_file "config/spring19_seminar/colapair_bert_frozen.conf" --override "exp_name = npi_bertmlm, run_name = run_bertmlm"