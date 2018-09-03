#!/bin/bash

# Master script to start a full suite of edge probing experiments on
# on Kubernetes. Comment out lines below to run only a subset of experiments.

PATH_TO_JIANT=${1:-"/nfs/jsalt/home/iftenney/jiant_exp"}
NOTIFY_EMAIL=${2:-"iftenney@gmail.com"}
MODE="create"
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

kuberun spr2-chars "elmo_chars_exp edges-spr2"
kuberun spr2-full "elmo_full_exp edges-spr2"
kuberun spr2-glove "glove_exp edges-spr2"
kuberun spr2-cove "cove_exp edges-spr2"
