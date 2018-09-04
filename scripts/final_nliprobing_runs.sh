#!/bin/bash

MODEL_DIR=$1 # directory of checkpoint to probe, e.g: /nfs/jsalt/share/models_to_probe/nli_do2_noelmo
EXP_NAME=$2
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"

# Note: you should only be overriding run_name and eval_tasks and (maybe) something like probe_path. 

# (Najoung) Prepositions, Negations
python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = prepswap, eval_tasks = nli-prob-prepswap, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, nli-prob-prepswap_use_classifier=mnli"
python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = negation, eval_tasks = nli-prob-negation, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, nli-prob-negation_use_classifier=mnli"

# (Tom) NPs 
python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = nps_final, eval_tasks = nps, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, nps_use_classifier=mnli"

# (Roma) Spatial, Quantifiers, Appearence, Comparators
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/spatial, nli-alt_use_classifier=mnli, nli-prob {probe_path = /nfs/jsalt/share/glue_data/roma-probing/probing/NLI-Prob/spatial.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/quant, nli-alt_use_classifier=mnli, nli-prob {probe_path = /nfs/jsalt/share/glue_data/roma-probing/probing/NLI-Prob/quant.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/appear, nli-alt_use_classifier=mnli, nli-prob {probe_path = /nfs/jsalt/share/glue_data/roma-probing/probing/NLI-Prob/appear.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/compare, nli-alt_use_classifier=mnli, nli-prob {probe_path = /nfs/jsalt/share/glue_data/roma-probing/probing/NLI-Prob/compare.tsv}"

# (Alexis) Implicatives, Factives, Neutrals
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/implicatives, nli-alt_use_classifier=mnli, nli-prob {probe_path = /nfs/jsalt/exp/alexis-probing/results/implicatives.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/factives, nli-alt_use_classifier=mnli, nli-prob {probe_path=/nfs/jsalt/exp/alexis-probing/results/factives.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = , eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}/neutrals, nli-alt_use_classifier=mnli, nli-prob {probe_path = /nfs/jsalt/exp/alexis-probing/results/neutrals.tsv}"

# (Adam) VerbNet, NER, Factuality, KG/Relation Extraction
python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name=recast-verbnet, eval_tasks=recast-verbnet, load_eval_checkpoint=${MODEL_FILE}, exp_name=${EXP_NAME}, train_for_eval=1, ${PROBING_TASK}_use_classifier=recast-verbnet"
python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name=recast-ner, eval_tasks=recast-ner, load_eval_checkpoint=${MODEL_FILE}, exp_name=${EXP_NAME}, train_for_eval=1, ${PROBING_TASK}_use_classifier=recast-ner"
python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name=recast-factuality, eval_tasks=recast-factuality, load_eval_checkpoint=${MODEL_FILE}, exp_name=${EXP_NAME}, train_for_eval=1, ${PROBING_TASK}_use_classifier=recast-factuality"
python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name=recast-kg, eval_tasks=recast-kg, load_eval_checkpoint=${MODEL_FILE}, exp_name=${EXP_NAME}, train_for_eval=1, ${PROBING_TASK}_use_classifier=recast-kg"
