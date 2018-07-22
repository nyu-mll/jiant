#!/bin/bash

# Script to run edge probing on ELMo-only.
# Because we learn a single set of ELMo weights during initial training,
# we do this as a series of sequential runs, with one task per run.

function run_exp() {
    OVERRIDES="exp_name=edgeprobe-bare-chars-$1, run_name=$1"
    OVERRIDES+=", train_tasks=$1"
    python main.py --config_file config/edgeprobe_bare.conf \
        -o "${OVERRIDES}" \
        --remote_log --notify iftenney@gmail.com
}

set -eux

cd $(dirname $0)
pushd "${PWD%jiant*}/jiant"

# run_exp "edges-srl-conll2005"
run_exp "edges-spr2"
run_exp "edges-dpr"
run_exp "edges-coref-ontonotes"
run_exp "edges-dep-labeling"

