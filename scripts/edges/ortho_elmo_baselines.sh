#!/bin/bash

# Script to run edge probing on ELMo-only.
# Run as separate experiments, since datasets are disjoint anyway.

NOTIFY_EMAIL=$1

function run_exp() {
    OVERRIDES="exp_name=elmo-ortho-$1, run_name=run_seed_$2"
    OVERRIDES+=", pretrain_tasks=$1, elmo_chars_only=0"
    OVERRIDES+=", target_tasks=$1, elmo_weight_file_path=/nfs/jsalt/home/berlin/elmo_2x4096_512_2048cnn_2xhighway_weights_ortho_seed_$2.hdf5"
    python main.py --config_file config/edgeprobe_bare.conf \
        -o "${OVERRIDES}" \
        --remote_log --notify "$NOTIFY_EMAIL"
}

set -eux

cd $(dirname $0)
pushd "${PWD%jiant*}/jiant"

run_exp "edges-srl-conll2005" "0"
run_exp "edges-srl-conll2005" "1"
run_exp "edges-srl-conll2005" "2"
run_exp "edges-spr2" "0"
run_exp "edges-spr2" "1"
run_exp "edges-spr2" "2"
run_exp "edges-dpr" "0"
run_exp "edges-dpr" "1"
run_exp "edges-dpr" "2"
run_exp "edges-coref-ontonotes" "0"
run_exp "edges-coref-ontonotes" "1"
run_exp "edges-coref-ontonotes" "2"
run_exp "edges-dep-labeling" "0"
run_exp "edges-dep-labeling" "1"
run_exp "edges-dep-labeling" "2"
run_exp "edges-ner-conll2003" "0"
run_exp "edges-ner-conll2003" "1"
run_exp "edges-ner-conll2003" "2"
run_exp "edges-constituent-ptb" "0"
run_exp "edges-constituent-ptb" "1"
run_exp "edges-constituent-ptb" "2"

#sudo poweroff
