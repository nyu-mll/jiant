#!/bin/bash

# Master script to start a full suite of edge probing experiments on
# on Kubernetes. Comment out lines below to run only a subset of experiments.
#
# See edgeprobe_exp_fns.sh for the override params, and config/edgeprobe_*.conf
# for the base configs.

MODE=${1:-"create"}
NOTIFY_EMAIL=${2:-"iftenney@gmail.com"}
PATH_TO_JIANT=${3:-"/nfs/jsalt/home/iftenney/jiant_exp"}
GPU_TYPE="p100"

function make_kubernetes_command() {
    # Generate shell command to execute in container.
    # Uses edgeprobe_exp_fns.sh to generate configs; see that file for details
    # and to define new experiments.
    echo -n "export NOTIFY_EMAIL=${NOTIFY_EMAIL}"
    echo -n "; pushd ${PATH_TO_JIANT}"
    echo -n "; source scripts/edgeprobe_runs/edgeprobe_exp_fns.sh"
    echo -n "; $@"
}

function kuberun() {
    # Create a Kubernetes job using run_batch.sh
    NAME=$1
    COMMAND=$(make_kubernetes_command $2)
    echo "Job '$NAME': '$COMMAND'"
    ./gcp/kubernetes/run_batch.sh $NAME "$COMMAND" $MODE $GPU_TYPE
    echo ""
}

##
# All experiments below.
# Uncomment the lines you want to run, or comment out those you don't.
##

task="dep-labeling-ewt"
kuberun elmo-chars-$task "elmo_chars_exp edges-$task"
kuberun elmo-ortho-$task "elmo_ortho_exp edges-$task 0"
kuberun elmo-full-$task  "elmo_full_exp edges-$task"
kuberun glove-$task      "glove_exp edges-$task"
kuberun cove-$task       "cove_exp edges-$task"

task="constituent-ontonotes"
kuberun elmo-chars-$task "elmo_chars_exp edges-$task"
kuberun elmo-ortho-$task "elmo_ortho_exp edges-$task 0"
kuberun elmo-full-$task  "elmo_full_exp edges-$task"
kuberun glove-$task      "glove_exp edges-$task"
kuberun cove-$task       "cove_exp edges-$task"

task="ner-ontonotes"
kuberun elmo-chars-$task "elmo_chars_exp edges-$task"
kuberun elmo-ortho-$task "elmo_ortho_exp edges-$task 0"
kuberun elmo-full-$task  "elmo_full_exp edges-$task"
kuberun glove-$task      "glove_exp edges-$task"
kuberun cove-$task       "cove_exp edges-$task"

task="srl-conll2012"
kuberun elmo-chars-$task "elmo_chars_exp edges-$task"
kuberun elmo-ortho-$task "elmo_ortho_exp edges-$task 0"
kuberun elmo-full-$task  "elmo_full_exp edges-$task"
kuberun glove-$task      "glove_exp edges-$task"
kuberun cove-$task       "cove_exp edges-$task"

task="coref-ontonotes-conll"
kuberun elmo-chars-$task "elmo_chars_exp edges-$task"
kuberun elmo-ortho-$task "elmo_ortho_exp edges-$task 0"
kuberun elmo-full-$task  "elmo_full_exp edges-$task"
kuberun glove-$task      "glove_exp edges-$task"
kuberun cove-$task       "cove_exp edges-$task"

task="spr2"
kuberun elmo-chars-$task "elmo_chars_exp edges-$task"
kuberun elmo-ortho-$task "elmo_ortho_exp edges-$task 0"
kuberun elmo-full-$task  "elmo_full_exp edges-$task"
kuberun glove-$task      "glove_exp edges-$task"
kuberun cove-$task       "cove_exp edges-$task"

task="dpr"
kuberun elmo-chars-$task "elmo_chars_exp edges-$task"
kuberun elmo-ortho-$task "elmo_ortho_exp edges-$task 0"
kuberun elmo-full-$task  "elmo_full_exp edges-$task"
kuberun glove-$task      "glove_exp edges-$task"
kuberun cove-$task       "cove_exp edges-$task"
