#!/bin/bash

# Script to run edge probing on ELMo-only.
# Run as separate experiments, since datasets are disjoint anyway.

function run_exp() {
    OVERRIDES="exp_name=elmo-ortho-$1, run_name=run"
    OVERRIDES+=", train_tasks=$1, elmo_chars_only=0"
    OVERRIDES+=", eval_tasks=$1, ortho_elmo=1"
    python main.py --config_file config/edgeprobe_bare.conf \
        -o "${OVERRIDES}" \
        --remote_log --notify bchen6@swarthmore.edu
}

#set -eux

#cd $(dirname $0)
#pushd "${PWD%jiant*}/jiant"

cd ..

run_exp "edges-srl-conll2005"
run_exp "edges-spr2"
run_exp "edges-dpr"
run_exp "edges-coref-ontonotes"
run_exp "edges-dep-labeling"
run_exp "edges-ner-conll2003"
run_exp "edges-constituent-ptb"

#sudo poweroff
