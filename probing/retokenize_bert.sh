#!/bin/bash

set -eu

EDGE_DATA_PATH="$JIANT_DATA_DIR/edges"
echo "Processing edge probing data in $EDGE_DATA_PATH"

declare -a SUBPATHS
SUBPATHS+=( "spr1" )
SUBPATHS+=( "spr2" )
SUBPATHS+=( "dpr" )
SUBPATHS+=( "dep_ewt" )
SUBPATHS+=( "ontonotes/const/pos" )
SUBPATHS+=( "ontonotes/const/nonterminal" )
SUBPATHS+=( "ontonotes/srl" )
SUBPATHS+=( "ontonotes/ner" )
SUBPATHS+=( "ontonotes/coref" )
SUBPATHS+=( "semeval" )
SUBPATHS+=( "tacred/rel" )
SUBPATHS+=( "noun_verb" )

for subpath in "${SUBPATHS[@]}"; do
  python $(dirname $0)/retokenize_edge_data.py \
    -t bert-base-uncased $EDGE_DATA_PATH/$subpath/*.json &
  python $(dirname $0)/retokenize_edge_data.py \
    -t bert-large-uncased $EDGE_DATA_PATH/$subpath/*.json &
done

# exit 0

# Only use the cased model on NER, per https://arxiv.org/pdf/1810.04805.pdf
CASED_SUBPATHS=( "ontonotes/ner" )

for subpath in "${CASED_SUBPATHS[@]}"; do
  python $(dirname $0)/retokenize_edge_data.py \
    -t bert-base-cased $EDGE_DATA_PATH/$subpath/*.json &
  python $(dirname $0)/retokenize_edge_data.py \
    -t bert-large-cased $EDGE_DATA_PATH/$subpath/*.json &
done

