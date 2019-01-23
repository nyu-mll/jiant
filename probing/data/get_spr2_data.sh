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

  # Univeral Dependencies 1.2 for English Web Treebank (ewt) source text.
  wget https://github.com/UniversalDependencies/UD_English/archive/r1.2.tar.gz
  mkdir ud
  tar -zxvf r1.2.tar.gz -C ud

  # Semantic Proto Roles annotations.
  wget http://decomp.io/projects/semantic-proto-roles/protoroles_eng_udewt.tar.gz
  mkdir protoroles
  tar -xvzf protoroles_eng_udewt.tar.gz -C protoroles

  popd
}

fetch_data

# Join UD with protorole annotations.
python $THIS_DIR/convert-spr2.py --src_dir $TARGET_DIR/raw -o $TARGET_DIR

# Print dataset stats for sanity-check.
python ${THIS_DIR%jiant*}/jiant/probing/edge_data_stats.py -i $TARGET_DIR/*.json
