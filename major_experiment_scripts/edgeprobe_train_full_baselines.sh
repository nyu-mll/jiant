#!/bin/bash

# Script to run edge probing on ELMo char-CNN only.
# Run as separate experiments, since datasets are disjoint anyway.

NOTIFY_EMAIL=$1

function run_exp() {
    OVERRIDES="exp_name=train-full-$1, run_name=run"
    OVERRIDES+=", elmo_chars_only=0"
    OVERRIDES+=", train_tasks=$1, max_vals=$2, val_interval=$3"
    python main.py --config_file config/edgeprobe_train.conf \
        -o "${OVERRIDES}" \
        --remote_log --notify "$NOTIFY_EMAIL"
}

set -eux

cd $(dirname $0)
pushd "${PWD%jiant*}/jiant"

run_exp "edges-srl-conll2005" 200 500
run_exp "edges-spr2" 100 100
run_exp "edges-dpr" 100 100
run_exp "edges-coref-ontonotes" 200 500
run_exp "edges-dep-labeling" 200 500

run_exp "edges-ner-conll2003" 200 250
run_exp "edges-constituent-ptb" 200 500
