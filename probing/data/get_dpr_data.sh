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

  # DPR data, as part of diverse NLI corpus (DNC),
  # a.k.a. "Inference is Everything"
  wget https://github.com/decompositional-semantics-initiative/DNC/raw/master/inference_is_everything.zip
  unzip inference_is_everything

  popd
}

fetch_data

# Convert DPR to edge probing JSON format.
python $THIS_DIR/convert-dpr.py --src_dir $TARGET_DIR/raw -o $TARGET_DIR

# Print dataset stats for sanity-check.
python ${THIS_DIR%jiant*}/jiant/probing/edge_data_stats.py -i $TARGET_DIR/*.json

