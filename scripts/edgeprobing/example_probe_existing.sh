#!/bin/bash

# Script to run an edge-probing task on an existing trained model.
# Based on prob_example_run.sh

# NOTE: don't be startled if you see a lot of warnings about missing parameters,
# like:
#    Parameter missing from checkpoint: edges-srl-conll2005_mdl.proj2.weight
# This is normal, because the probing task won't be in the original checkpoint.

MODEL_DIR=$1 # directory of checkpoint to probe,
             # e.g: /nfs/jsalt/share/models_to_probe/nli_do2_noelmo
PROBING_TASK=${2:-"edges-spr2"}  # probing task name(s)

EXP_NAME=${3:-"edgeprobe-$(basename $MODEL_DIR)"}  # experiment name
RUN_NAME=${4:-"probing"}                     # name for this run

PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}"

pushd "${PWD%jiant*}jiant"

# Load defaults.conf for any missing params, then model param file,
# then eval_existing.conf to override paths & eval config.
# Finally, apply custom overrides defined above.
# To add email notifications, add an additional argument:
#   --notify my_email@example.com
python main.py -c jiant/config/defaults.conf ${PARAM_FILE} jiant/config/edgeprobe/edgeprobe_existing.conf \
    -o "${OVERRIDES}" --remote_log
