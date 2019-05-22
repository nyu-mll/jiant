#!/bin/bash

# Script to run edge probing on ELMo char-CNN only.
# Run as separate experiments, since datasets are disjoint anyway.

NOTIFY_EMAIL=$1

function run_exp() {
    OVERRIDES="exp_name=train-chars-$1, run_name=run"
    OVERRIDES+=", pretrain_tasks=$1, max_vals=$2, val_interval=$3"
    python main.py --config_file config/edgeprobe/edgeprobe_train.conf \
        -o "${OVERRIDES}" \
        --remote_log --notify "$NOTIFY_EMAIL"
}

set -eux

cd $(dirname $0)
pushd "${PWD%jiant*}/jiant"

# Small tasks
run_exp "edges-spr2" 100 100
run_exp "edges-dpr" 100 100
run_exp "edges-dep-labeling" 200 500
run_exp "edges-ner-conll2003" 200 250

# OntoNotes
run_exp "edges-srl-conll2012" 200 1000
run_exp "edges-coref-ontonotes-conll" 200 1000
run_exp "edges-ner-ontonotes" 200 1000
run_exp "edges-constituent-ontonotes" 200 1000

# run_exp "edges-srl-conll2005"
# run_exp "edges-coref-ontonotes"
# run_exp "edges-constituent-ptb"
# run_exp "edges-ccg-tag"
