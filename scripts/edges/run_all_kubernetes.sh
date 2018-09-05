#!/bin/bash

# Master script to start a full suite of edge probing experiments on
# on Kubernetes. Comment out lines below to run only a subset of experiments.
#
# See exp_fns.sh for the override params, and config/edgeprobe_*.conf
# for the base configs.

NOTIFY_EMAIL="iftenney@gmail.com"
GPU_TYPE="p100"
PROJECT="edgeprobe"
PATH_TO_JIANT="/nfs/jsalt/home/iftenney/jiant_exp"

# Handle flags.
OPTIND=1         # Reset in case getopts has been used previously in the shell.
while getopts ":g:p:n:j:" opt; do
    case "$opt" in
    g)  GPU_TYPE=$OPTARG
        ;;
    p)  PROJECT=$OPTARG
        ;;
    n)	NOTIFY_EMAIL=$OPTARG
        ;;
    j)  PATH_TO_JIANT=$OPTARG
        ;;
    \? )
        echo "Invalid flag $opt."
        exit 1
        ;;
    esac
done
shift $((OPTIND-1))

# Remaining positional arguments.
MODE=$1

function make_kubernetes_command() {
    # Generate shell command to execute in container.
    # Uses edgeprobe_exp_fns.sh to generate configs; see that file for details
    # and to define new experiments.
    echo -n "export NOTIFY_EMAIL=${NOTIFY_EMAIL}"
    echo -n "; pushd ${PATH_TO_JIANT}"
    echo -n "; source scripts/edges/exp_fns.sh"
    echo -n "; $@"
}

function kuberun() {
    # Create a Kubernetes job using run_batch.sh
    NAME=$1
    COMMAND=$(make_kubernetes_command $2)
    echo "Job '$NAME': '$COMMAND'"
    ./gcp/kubernetes/run_batch.sh -m $MODE -p ${PROJECT} -g ${GPU_TYPE} \
        $NAME "$COMMAND"
    echo ""
}

##
# All experiments below.
# Uncomment the lines you want to run, or comment out those you don't.
##

declare -a ALL_TASKS
ALL_TASKS+=( "dep-labeling-ewt" )
ALL_TASKS+=( "constituent-ontonotes" )
ALL_TASKS+=( "ner-ontonotes" )
ALL_TASKS+=( "srl-conll2012" )
ALL_TASKS+=( "coref-ontonotes-conll" )
ALL_TASKS+=( "spr2" )
ALL_TASKS+=( "dpr" )
echo "All tasks to run: ${ALL_TASKS[@]}"

for task in "${ALL_TASKS[@]}"
do
  kuberun elmo-chars-$task "elmo_chars_exp edges-$task"
  kuberun elmo-ortho-$task "elmo_ortho_exp edges-$task 0"
  kuberun elmo-full-$task  "elmo_full_exp edges-$task"
  kuberun glove-$task      "glove_exp edges-$task"
  kuberun cove-$task       "cove_exp edges-$task"
done

