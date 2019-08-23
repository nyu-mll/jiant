#!/bin/bash

# Library of experiment functions for running main edge-probing experiments.
# Don't execute this script directly, but instead include at the top of an
# experiment script as:
#
#   export NOTIFY_EMAIL="yourname@gmail.com"  # optional, only works with creds
#   pushd /path/to/jiant
#   source scripts/edges/exp_fns.sh
#   elmo_chars_exp edges-srl-conll2012
#   elmo_full_exp edges-srl-conll2012
#   elmo_ortho_exp edges-srl-conll2012 0
#
#
# See individual functions below for usage.

EP_RESOURCE_DIR="/nfs/jiant/share/edge-probing/resources"

function run_exp() {
    # Helper function to invoke main.py.
    # Don't run this directly - use the experiment functions below,
    # or create a new one for a new experiment suite.
    # Usage: run_exp <config_file> <overrides>
    CONFIG_FILE=$1
    OVERRIDES=$2
    declare -a args
    args+=( --config_file "${CONFIG_FILE}" )
    args+=( -o "${OVERRIDES}" --remote_log )
    if [ ! -z $NOTIFY_EMAIL ]; then
        args+=( --notify "$NOTIFY_EMAIL" )
    fi
    python main.py "${args[@]}"
}

function elmo_chars_exp() {
    # Lexical baseline, probe ELMo char CNN layer.
    # Usage: elmo_chars_exp <task_name>
    OVERRIDES="exp_name=elmo-chars-$1, run_name=run"
    OVERRIDES+=", target_tasks=$1, input_module=elmo-chars-only"
    run_exp "jiant/config/edgeprobe/edgeprobe_bare.conf" "${OVERRIDES}"
}

function elmo_full_exp() {
    # Full ELMo, probe full ELMo with learned mixing weights.
    # Usage: elmo_full_exp <task_name>
    OVERRIDES="exp_name=elmo-full-$1, run_name=run"
    OVERRIDES+=", target_tasks=$1, input_module=elmo"
    run_exp "jiant/config/edgeprobe/edgeprobe_bare.conf" "${OVERRIDES}"
}

function elmo_ortho_exp() {
    # Full ELMo with random orthogonal weights for LSTM and projections.
    # Usage: elmo_ortho_exp <task_name> <random_seed>
    ELMO_WEIGHTS_PATH="${EP_RESOURCE_DIR}/random_elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_ortho_seed_$2.hdf5"
    OVERRIDES="exp_name=elmo-ortho-$1, run_name=run_seed_$2"
    OVERRIDES+=", target_tasks=$1, input_module=elmo"
    OVERRIDES+=", elmo_weight_file_path=${ELMO_WEIGHTS_PATH}"
    run_exp "jiant/config/edgeprobe/edgeprobe_bare.conf" "${OVERRIDES}"
}

function elmo_random_exp() {
    # Full ELMo with random normal weights for LSTM and projections.
    # Usage: elmo_random_exp <task_name> <random_seed>
    ELMO_WEIGHTS_PATH="${EP_RESOURCE_DIR}/random_elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_random_seed_$2.hdf5"
    OVERRIDES="exp_name=elmo-random-$1, run_name=run_seed_$2"
    OVERRIDES+=", target_tasks=$1, input_module=elmo"
    OVERRIDES+=", elmo_weight_file_path=${ELMO_WEIGHTS_PATH}"
    run_exp "jiant/config/edgeprobe/edgeprobe_bare.conf" "${OVERRIDES}"
}

function train_chars_exp() {
    # Trained encoder over ELMo character layer.
    # Usage: train_chars_exp <task_name> <max_vals> <val_interval>
    OVERRIDES="exp_name=train-chars-$1, run_name=run"
    OVERRIDES+=", pretrain_tasks=$1, max_vals=$2, val_interval=$3"
    OVERRIDES+=", input_module=elmo-chars-only"
    run_exp "jiant/config/edgeprobe/edgeprobe_train.conf" "${OVERRIDES}"
}

function train_full_exp() {
    # Trained encoder over full ELMo.
    # Usage: train_full_exp <task_name> <max_vals> <val_interval>
    OVERRIDES="exp_name=train-full-$1, run_name=run"
    OVERRIDES+=", pretrain_tasks=$1, max_vals=$2, val_interval=$3"
    OVERRIDES+=", input_module=elmo"
    run_exp "jiant/config/edgeprobe/edgeprobe_train.conf" "${OVERRIDES}"
}

##
# GloVe and CoVe-based models.
function glove_exp() {
    # Lexical baseline, probe GloVe embeddings.
    # Usage: glove_exp <task_name>
    OVERRIDES="exp_name=glove-$1, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    run_exp "jiant/config/edgeprobe/edgeprobe_glove.conf" "${OVERRIDES}"
}

function cove_exp() {
    # Probe CoVe, which is concatenated with GloVe per standard usage.
    # Usage: cove_exp <task_name>
    OVERRIDES="exp_name=cove-$1, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    run_exp "jiant/config/edgeprobe/edgeprobe_cove.conf" "${OVERRIDES}"
}

##
# OpenAI transformer model.
function openai_exp() {
    # Probe the OpenAI transformer model.
    # Usage: openai_exp <task_name>
    OVERRIDES="exp_name=openai-$1, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    run_exp "jiant/config/edgeprobe/edgeprobe_openai.conf" "${OVERRIDES}"
}

function openai_cat_exp() {
    # As above, but concat embeddings to output.
    # Usage: openai_cat_exp <task_name>
    OVERRIDES="exp_name=openai-cat-$1, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", openai_output_mode=cat"
    run_exp "jiant/config/edgeprobe/edgeprobe_openai.conf" "${OVERRIDES}"
}

function openai_lex_exp() {
    # Probe the OpenAI transformer model base layer.
    # Usage: openai_lex_exp <task_name>
    OVERRIDES="exp_name=openai-lex-$1, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", openai_output_mode=only"
    run_exp "jiant/config/edgeprobe/edgeprobe_openai.conf" "${OVERRIDES}"
}

function openai_mix_exp() {
    # Probe the OpenAI transformer with ELMo-style scalar mixing across layers.
    # Usage: openai_mix_exp <task_name>
    OVERRIDES="exp_name=openai-mix-$1, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", openai_output_mode=mix"
    run_exp "jiant/config/edgeprobe/edgeprobe_openai.conf" "${OVERRIDES}"
}

function openai_bwb_exp() {
    # Probe the OpenAI transformer model, as trained on BWB-shuffled.
    # Usage: openai_bwb_exp <task_name>
    CKPT_PATH="${EP_RESOURCE_DIR}/checkpoints/bwb_shuffled/model.ckpt-1000000"
    OVERRIDES="exp_name=openai-bwb-$1, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", openai_transformer_ckpt=${CKPT_PATH}"
    OVERRIDES+=", openai_output_mode=cat"
    run_exp "jiant/config/edgeprobe/edgeprobe_openai.conf" "${OVERRIDES}"
}

##
# BERT model.
# These take two arguments: the task name, and the bert model
# (e.g. base-uncased)
function bert_cat_exp() {
    # Run BERT, and concat embeddings to output.
    # Usage: bert_cat_exp <task_name> <bert_model_name>
    OVERRIDES="exp_name=bert-${2}-cat-${1}, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", input_module=bert-$2"
    OVERRIDES+=", pytorch_transformers_output_mode=cat"
    run_exp "jiant/config/edgeprobe/edgeprobe_bert.conf" "${OVERRIDES}"
}

function bert_lex_exp() {
    # Probe the BERT token embeddings.
    # Usage: bert_lex_exp <task_name> <bert_model_name>
    OVERRIDES="exp_name=bert-${2}-lex-${1}, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", input_module=bert-$2"
    OVERRIDES+=", pytorch_transformers_output_mode=only"
    run_exp "jiant/config/edgeprobe/edgeprobe_bert.conf" "${OVERRIDES}"
}

function bert_mix_exp() {
    # Run BERT with ELMo-style scalar mixing across layers.
    # Usage: bert_mix_exp <task_name> <bert_model_name>
    OVERRIDES="exp_name=bert-${2}-mix-${1}, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", input_module=bert-$2"
    OVERRIDES+=", pytorch_transformers_output_mode=mix"
    run_exp "jiant/config/edgeprobe/edgeprobe_bert.conf" "${OVERRIDES}"
}

##
# BERT layer experiments, for ACL paper
function bert_mix_k_exp() {
    # Run BERT with ELMo-style scalar mixing across the first K layers.
    # Usage: bert_mix_k_exp <task_name> <bert_model_name> <k>
    OVERRIDES="exp_name=bert-${2}-mix_${3}-${1}, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", input_module=bert-$2"
    OVERRIDES+=", pytorch_transformers_output_mode=mix"
    OVERRIDES+=", pytorch_transformers_max_layer=${3}"
    run_exp "jiant/config/edgeprobe/edgeprobe_bert.conf" "${OVERRIDES}"
}

function bert_at_k_exp() {
    # Run BERT and probe layer K.
    # Usage: bert_at_k_exp <task_name> <bert_model_name> <k>
    OVERRIDES="exp_name=bert-${2}-at_${3}-${1}, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", input_module=bert-$2"
    OVERRIDES+=", pytorch_transformers_output_mode=top"
    OVERRIDES+=", pytorch_transformers_max_layer=${3}"
    run_exp "jiant/config/edgeprobe/edgeprobe_bert.conf" "${OVERRIDES}"
}
