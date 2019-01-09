#!/bin/bash

set -eu

EDGE_DATA_PATH="/nfs/jsalt/share/glue_data/edges"

declare -a SUBPATHS
SUBPATHS+=( "ontonotes-constituents" )
SUBPATHS+=( "dep_ewt" )
SUBPATHS+=( "ontonotes-ner" )
SUBPATHS+=( "srl_conll2012" )
SUBPATHS+=( "ontonotes-coref-conll" )
SUBPATHS+=( "spr1" )
SUBPATHS+=( "spr2" )
SUBPATHS+=( "dpr" )

for subpath in "${SUBPATHS[@]}"; do
  python $(dirname $0)/retokenize_edge_data.bert.py \
    --model bert-base-uncased $EDGE_DATA_PATH/$subpath/*.json &
  python $(dirname $0)/retokenize_edge_data.bert.py \
    --model bert-large-uncased $EDGE_DATA_PATH/$subpath/*.json &
done

# exit 0

# Only use the cased model on NER, per https://arxiv.org/pdf/1810.04805.pdf
CASED_SUBPATHS=( "ontonotes-ner" )

for subpath in "${CASED_SUBPATHS[@]}"; do
  python $(dirname $0)/retokenize_edge_data.bert.py \
    --model bert-base-cased $EDGE_DATA_PATH/$subpath/*.json &
  python $(dirname $0)/retokenize_edge_data.bert.py \
    --model bert-large-cased $EDGE_DATA_PATH/$subpath/*.json &
done

