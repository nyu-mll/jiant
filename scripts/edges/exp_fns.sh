#!/bin/bash

# Library of experiment functions for running main edge-probing experiments.
# Don't execute this script directly, but instead include at the top of an
# experiment script as:
#
#   export NOTIFY_EMAIL="yourname@gmail.com"
#   pushd /path/to/jiant
#   source scripts/edges/exp_fns.sh
#   elmo_chars_exp edges-srl-conll2005
#   elmo_full_exp edges-srl-conll2005
#   elmo_ortho_exp edges-srl-conll2005 0
#
#
# See individual functions below for usage.

function run_exp() {
    # Helper function to invoke main.py.
    # Don't run this directly - use the experiment functions below,
    # or create a new one for a new experiment suite.
    # Usage: run_exp <config_file> <overrides>
    CONFIG_FILE=$1
    OVERRIDES=$2
    python main.py --config_file "${CONFIG_FILE}" \
        -o "${OVERRIDES}" --remote_log --notify "$NOTIFY_EMAIL"
}

function elmo_chars_exp() {
    # Lexical baseline, probe ELMo char CNN layer.
    # Usage: elmo_chars_exp <task_name>
    OVERRIDES="exp_name=elmo-chars-$1, run_name=run"
    OVERRIDES+=", eval_tasks=$1"
    run_exp "config/edgeprobe_bare.conf" "${OVERRIDES}"
}

function elmo_full_exp() {
    # Full ELMo, probe full ELMo with learned mixing weights.
    # Usage: elmo_full_exp <task_name>
    OVERRIDES="exp_name=elmo-full-$1, run_name=run"
    OVERRIDES+=", eval_tasks=$1, elmo_chars_only=0"
    run_exp "config/edgeprobe_bare.conf" "${OVERRIDES}"
}

function elmo_ortho_exp() {
    # Full ELMo with random orthogonal weights for LSTM and projections.
    # Usage: elmo_ortho_exp <task_name> <random_seed>
    ELMO_WEIGHTS_PATH="/nfs/jsalt/share/random_elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_ortho_seed_$2.hdf5"
    OVERRIDES="exp_name=elmo-ortho-$1, run_name=run_seed_$2"
    OVERRIDES+=", eval_tasks=$1, elmo_chars_only=0"
    OVERRIDES+=", elmo_weight_file_path=${ELMO_WEIGHTS_PATH}"
    run_exp "config/edgeprobe_bare.conf" "${OVERRIDES}"
}

function elmo_random_exp() {
    # Full ELMo with random normal weights for LSTM and projections.
    # Usage: elmo_random_exp <task_name> <random_seed>
    ELMO_WEIGHTS_PATH="/nfs/jsalt/share/random_elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_random_seed_$2.hdf5"
    OVERRIDES="exp_name=elmo-random-$1, run_name=run_seed_$2"
    OVERRIDES+=", eval_tasks=$1, elmo_chars_only=0"
    OVERRIDES+=", elmo_weight_file_path=${ELMO_WEIGHTS_PATH}"
    run_exp "config/edgeprobe_bare.conf" "${OVERRIDES}"
}

function train_chars_exp() {
    # Trained encoder over ELMo character layer.
    # Usage: train_chars_exp <task_name> <max_vals> <val_interval>
    OVERRIDES="exp_name=train-chars-$1, run_name=run"
    OVERRIDES+=", train_tasks=$1, max_vals=$2, val_interval=$3"
    run_exp "config/edgeprobe_train.conf" "${OVERRIDES}"
}

function train_full_exp() {
    # Trained encoder over full ELMo.
    # Usage: train_full_exp <task_name> <max_vals> <val_interval>
    OVERRIDES="exp_name=train-full-$1, run_name=run"
    OVERRIDES+=", train_tasks=$1, max_vals=$2, val_interval=$3, elmo_chars_only=0"
    run_exp "config/edgeprobe_train.conf" "${OVERRIDES}"
}

##
# GloVe and CoVe-based models.
function glove_exp() {
    # Lexical baseline, probe GloVe embeddings.
    # Usage: glove_exp <task_name>
    OVERRIDES="exp_name=glove-$1, run_name=run"
    OVERRIDES+=", eval_tasks=$1"
    run_exp "config/edgeprobe_glove.conf" "${OVERRIDES}"
}

function cove_exp() {
    # Probe CoVe, which is concatenated with GloVe per standard usage.
    # Usage: cove_exp <task_name>
    OVERRIDES="exp_name=cove-$1, run_name=run"
    OVERRIDES+=", eval_tasks=$1"
    run_exp "config/edgeprobe_cove.conf" "${OVERRIDES}"
}
