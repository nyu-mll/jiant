#!/bin/bash

TARGET_DIR=$1

THIS_DIR=$(realpath $(dirname $0))

set -e
if [ ! -d $TARGET_DIR ]; then
  mkdir $TARGET_DIR
fi

function fetch_data() {
  mkdir -p $TARGET_DIR/raw
  pushd $TARGET_DIR/raw

  # Univeral Dependencies 2.2 for English Web Treebank (ewt) source text.
  wget https://github.com/UniversalDependencies/UD_English-EWT/archive/r2.2.tar.gz
  mkdir ud
  tar -zxvf r2.2.tar.gz -C ud

  # # Universal Dependencies v2.3 data
  # git clone https://github.com/UniversalDependencies/UD_English-EWT.git $TARGET_DIR/raw

  popd
}

fetch_data

# Convert UD to edge probing format.
python $THIS_DIR/ud_to_json.py \
  -i $TARGET_DIR/raw/ud/UD_English-EWT-r2.2/en_ewt-ud-*.conllu \
  -o $TARGET_DIR

# Print dataset stats for sanity-check.
python ${THIS_DIR%jiant*}/jiant/probing/edge_data_stats.py -i $TARGET_DIR/*.json
