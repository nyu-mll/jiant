#!/bin/bash

MODEL_DIR=$1 # directory of checkpoint to probe, e.g: /nfs/jsalt/share/models_to_probe/nli_do2_noelmo
PROBING_TASK=$2 # task name, e.g. recast-puns

EXP_NAME="probing"
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"

# Note: you should only be overriding run_name and eval_tasks and (maybe) something like probe_path. 

# Example of run that would probe on puns
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = recast-puns, eval_tasks = recast-puns, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"

# TODO(Najoung) Prepositions, Negations
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"

# TODO(Tom) NPs 
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"

# TODO(Roma) Spatial, Quantifiers, Appearence, Comparators
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = spatial, eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli, nli-alt {probe_path = /nfs/jsalt/home/roma/probing/NLI-Prob/spatial.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = quant, eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli, nli-alt {probe_path = /nfs/jsalt/home/roma/probing/NLI-Prob/quant.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = appear, eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli, nli-alt {probe_path = /nfs/jsalt/home/roma/probing/NLI-Prob/appear.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = compare, eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli, nli-alt {probe_path = /nfs/jsalt/home/roma/probing/NLI-Prob/compare.tsv}"

# TODO(Alexis) Implicatives, Factives, Neutrals
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = implicatives, eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli, nli-alt {probe_path = /nfs/jsalt/exp/alexis-probing/results/implicatives.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = factives, eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli, nli-alt {probe_path=/nfs/jsalt/exp/alexis-probing/results/factives.tsv}"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = neutrals, eval_tasks = nli-alt, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli, nli-alt {probe_path = /nfs/jsalt/exp/alexis-probing/results/neutrals.tsv}"
