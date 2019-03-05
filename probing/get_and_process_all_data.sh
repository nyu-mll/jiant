#!/bin/bash

# Master script to re-generate all edge probing datasets.
# This script just follows the instructions in data/README.md;
# see that file and the individual dataset scripts for more details.
#
# Usage:
#  # First, modify the paths below to point to OntoNotes and SPR1 on your system.
#  ./get_and_process_all_data.sh /path/to/glue_data
#
# Note that OntoNotes is rather large and we need to process it several times, so
# this script can take a while to run.

JIANT_DATA_DIR=${1:-"$HOME/glue_data"}  # path to glue_data directory

## Configure these for your environment ##
PATH_TO_ONTONOTES="/nfs/jsalt/home/iftenney/ontonotes/ontonotes/conll-formatted-ontonotes-5.0"
PATH_TO_SPR1_RUDINGER="/nfs/jsalt/home/iftenney/decomp.net/spr1"

## Don't modify below this line. ##

set -eux

OUTPUT_DIR="${JIANT_DATA_DIR}/edges"
mkdir -p $OUTPUT_DIR
HERE=$(dirname $0)

function preproc_task() {
    TASK_DIR=$1
    # Extract data labels.
    python $HERE/get_edge_data_labels.py -o $TASK_DIR/labels.txt \
      -i $TASK_DIR/*.json -s
    # Retokenize for each tokenizer we need.
    python $HERE/retokenize_edge_data.py -t "MosesTokenizer" $TASK_DIR/*.json
    python $HERE/retokenize_edge_data.py -t "OpenAI.BPE"     $TASK_DIR/*.json
    python $HERE/retokenize_edge_data.py -t "bert-base-uncased"  $TASK_DIR/*.json
    python $HERE/retokenize_edge_data.py -t "bert-large-uncased" $TASK_DIR/*.json

    # Convert the original version to tfrecord.
    python $HERE/convert_edge_data_to_tfrecord.py $TASK_DIR/*.json
}

function get_ontonotes() {
    ## OntoNotes (all tasks)
    ## Gives ontonotes/{task}/{split}.json,
    ## where task = {const, coref, ner, srl}
    ## and split = {train, development, test, conll-2012-test}
    # TODO: standardize filenames!
    python $HERE/data/extract_ontonotes_all.py --ontonotes "${PATH_TO_ONTONOTES}" \
        --tasks const coref ner srl \
        --splits train development test conll-2012-test \
        -o $OUTPUT_DIR/ontonotes

    ## Gives ontonotes/const/{pos,nonterminal}/{split}.json
    python $HERE/split_constituent_data.py $OUTPUT_DIR/ontonotes/const/*.json

    preproc_task $OUTPUT_DIR/ontonotes/const
    preproc_task $OUTPUT_DIR/ontonotes/const/pos
    preproc_task $OUTPUT_DIR/ontonotes/const/nonterminal
    preproc_task $OUTPUT_DIR/ontonotes/coref
    preproc_task $OUTPUT_DIR/ontonotes/ner
    preproc_task $OUTPUT_DIR/ontonotes/srl
}

function get_spr_dpr() {
    ## SPR1
    ## Gives spr1/spr1.{split}.json, where split = {train, dev, test}
    # TODO: standardize filenames!
    python $HERE/data/convert-spr1-rudinger.py \
        -i ${PATH_TO_SPR1_RUDINGER}/*.json \
        -o $OUTPUT_DIR/spr1
    preproc_task $OUTPUT_DIR/spr1

    ## SPR2
    ## Gives spr2/edges.{split}.json, where split = {train, dev, test}
    # TODO: standardize filenames!
    pip install conllu
    bash $HERE/data/get_spr2_data.sh $OUTPUT_DIR/spr2
    preproc_task $OUTPUT_DIR/spr2

    ## DPR
    ## Gives dpr/{split}.json, where split = {train, dev, test}
    bash $HERE/data/get_dpr_data.sh $OUTPUT_DIR/dpr
    preproc_task $OUTPUT_DIR/dpr
}

function get_ud() {
    ## Universal Dependencies
    ## Gives dep_ewt/en_ewt-ud-{split}.json, where split = {train, dev, test}
    # TODO: standardize filenames!
    bash $HERE/data/get_ud_data.sh $OUTPUT_DIR/dep_ewt
    preproc_task $OUTPUT_DIR/dep_ewt
}

function get_semeval() {
    ## SemEval 2010 Task 8 relation classification
    ## Gives semeval/{split}.json, where split = {train.0.85, dev, test}
    mkdir $OUTPUT_DIR/semeval
    bash $HERE/data/get_semeval_data.sh $OUTPUT_DIR/semeval
    preproc_task $OUTPUT_DIR/semeval
}

get_ontonotes
get_spr_dpr
get_ud

get_semeval

