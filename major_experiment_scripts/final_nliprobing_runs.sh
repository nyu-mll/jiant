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
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"

# TODO(Alexis) Implicatives, Factives, Neutrals
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"
python main.py -c config/defaults.conf ${PARAM_FILE} config/eval_existing.conf -o "run_name = ???, eval_tasks = ???, load_eval_checkpoint = ${MODEL_FILE}, exp_name = ${EXP_NAME}, ${PROBING_TASK}_use_classifier=mnli"
