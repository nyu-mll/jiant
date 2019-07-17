#!/bin/bash

# Master script to start a full suite of edge probing experiments on
# on Kubernetes. Comment out lines below to run only a subset of experiments.
#
# See exp_fns.sh for the override params, and config/edgeprobe/edgeprobe_*.conf
# for the base configs.
#
# To run on a particular cluster, authenticate to that cluster with:
#   gcloud container clusters get-credentials --zone <zone> <cluster_name>

set -e

# Default arguments.
PROJECT=""
NOTIFY_EMAIL="$NOTIFY_EMAIL"  # allow pre-set from shell

# Handle flags.
OPTIND=1         # Reset in case getopts has been used previously in the shell.
while getopts ":p:n:" opt; do
    case "$opt" in
    p)  PROJECT=$OPTARG
        ;;
    n)	NOTIFY_EMAIL=$OPTARG
        ;;
    \? )
        echo "Invalid flag $opt."
        exit 1
        ;;
    esac
done
shift $((OPTIND-1))

# Remaining positional arguments.
MODE=${1:-"create"}

if [ -z $PROJECT ]; then
    echo "You must provide a project name!"
    exit 1
fi

if [ -z $NOTIFY_EMAIL ]; then
    echo "You must provide an email address!"
    exit 1
fi

# Top-level directory for the current repo.
pushd $(git rev-parse --show-toplevel)

# Make a copy of the current tree in the project directory.
PROJECT_DIR="/nfs/jsalt/exp/$PROJECT"
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "Creating project directory ${PROJECT_DIR}"
    mkdir ${PROJECT_DIR}
    chmod -R o+w ${PROJECT_DIR}
fi
if [[ $MODE != "delete" ]]; then
    echo "Copying source tree to project dir..."
    rsync -a --exclude=".git" "./" "${PROJECT_DIR}/jiant"
fi
PATH_TO_JIANT="${PROJECT_DIR}/jiant"

function make_kubernetes_command() {
    # Generate shell command to execute in container.
    # Uses exp_fns.sh to generate configs; see that file for details
    # and to define new experiments.
    echo -n "pushd ${PATH_TO_JIANT}"
    echo -n "; source scripts/edges/exp_fns.sh"
    echo -n "; $@"
}

function kuberun() {
    # Create a Kubernetes job using run_batch.sh
    NAME=$1
    COMMAND=$(make_kubernetes_command $2)
    echo "Job '$NAME': '$COMMAND'"
    ./gcp/kubernetes/run_batch.sh -m $MODE -p ${PROJECT} -g ${GPU_TYPE} \
        -n ${NOTIFY_EMAIL} $NAME "$COMMAND"
    echo ""
}

##
# All experiments below.
# Uncomment the lines you want to run, or comment out those you don't.
##

declare -a ALL_TASKS
ALL_TASKS+=( "spr1" )
ALL_TASKS+=( "spr2" )
ALL_TASKS+=( "dpr" )
ALL_TASKS+=( "dep-ud-ewt" )
ALL_TASKS+=( "nonterminal-ontonotes" )
ALL_TASKS+=( "pos-ontonotes" )
ALL_TASKS+=( "ner-ontonotes" )
ALL_TASKS+=( "srl-ontonotes" )
ALL_TASKS+=( "coref-ontonotes" )
ALL_TASKS+=( "rel-semeval" )
echo "All tasks to run: ${ALL_TASKS[@]}"

if [[ $MODE == "delete" ]]; then
    # OK to fail to delete some jobs, still try others.
    set +e
fi

##
# Run these on the main 'jsalt' cluster
gcloud container clusters get-credentials --zone us-east1-c jsalt
export GPU_TYPE="p100"
for task in "${ALL_TASKS[@]}"
do
    kuberun elmo-chars-$task "elmo_chars_exp edges-$task"
    kuberun elmo-ortho-$task "elmo_ortho_exp edges-$task 0"
    kuberun elmo-full-$task  "elmo_full_exp edges-$task"
    kuberun glove-$task      "glove_exp edges-$task"
    kuberun cove-$task       "cove_exp edges-$task"

    kuberun openai-lex-$task "openai_lex_exp edges-$task"
    kuberun bert-base-uncased-lex-$task    "bert_lex_exp edges-$task base-uncased"
    kuberun bert-large-uncased-lex-$task   "bert_lex_exp edges-$task large-uncased"
done

##
# Run these on 'jsalt-central' for V100s
gcloud container clusters get-credentials --zone us-central1-a jsalt-central
export GPU_TYPE="v100"
for task in "${ALL_TASKS[@]}"
do
    kuberun openai-cat-$task "openai_cat_exp edges-$task"
    kuberun openai-mix-$task "openai_mix_exp edges-$task"
    kuberun openai-bwb-$task "openai_bwb_exp edges-$task"

    kuberun bert-base-uncased-cat-$task    "bert_cat_exp edges-$task base-uncased"
    kuberun bert-large-uncased-cat-$task   "bert_cat_exp edges-$task large-uncased"

    # BERT with ELMo-style scalar mixing.
    kuberun bert-base-uncased-mix-$task    "bert_mix_exp edges-$task base-uncased"
    kuberun bert-large-uncased-mix-$task   "bert_mix_exp edges-$task large-uncased"
done

