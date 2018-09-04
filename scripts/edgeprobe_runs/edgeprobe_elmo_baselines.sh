#!/bin/bash

# Script to run edge probing on ELMo char-CNN only.
# Run as separate experiments, since datasets are disjoint anyway.

NOTIFY_EMAIL=$1

function run_exp() {
    OVERRIDES="exp_name=elmo-chars-$1, run_name=run"
    OVERRIDES+=", eval_tasks=$1"
    python main.py --config_file config/edgeprobe_bare.conf \
        -o "${OVERRIDES}" \
        --remote_log --notify "$NOTIFY_EMAIL"
}

set -eux

cd $(dirname $0)
pushd "${PWD%jiant*}/jiant"

# Small tasks
run_exp "edges-spr2"
run_exp "edges-dpr"
run_exp "edges-dep-labeling"
run_exp "edges-ner-conll2003"

# OntoNotes
run_exp "edges-srl-conll2012"
run_exp "edges-coref-ontonotes-conll"
run_exp "edges-ner-ontonotes"
run_exp "edges-constituent-ontonotes"

# run_exp "edges-srl-conll2005"
# run_exp "edges-coref-ontonotes"
# run_exp "edges-constituent-ptb"
# run_exp "edges-ccg-tag"
