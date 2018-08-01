#!/bin/bash

# Script to run edge probing on ELMo-only.
# Run as separate experiments, since datasets are disjoint anyway.

NOTIFY_EMAIL=$1

function run_exp() {
    for i in {0...2}
    do
        OVERRIDES="exp_name=elmo-ortho-$1, run_name=run_seed_$i"
        OVERRIDES+=", train_tasks=$1, elmo_chars_only=0"
        OVERRIDES+=", eval_tasks=$1, elmo_weight_file_path=/nfs/jsalt/home/berlin/elmo_2x4096_512_2048cnn_2xhighway_weights_ortho_seed_$i"
        python main.py --config_file config/edgeprobe_bare.conf \
            -o "${OVERRIDES}" \
            --remote_log --notify "$NOTIFY_EMAIL"
    done
}

set -eux

cd $(dirname $0)
pushd "${PWD%jiant*}/jiant"

run_exp "edges-srl-conll2005"
run_exp "edges-spr2"
run_exp "edges-dpr"
run_exp "edges-coref-ontonotes"
run_exp "edges-dep-labeling"
run_exp "edges-ner-conll2003"
run_exp "edges-constituent-ptb"

#sudo poweroff
