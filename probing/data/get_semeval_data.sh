#!/bin/bash

TARGET_DIR=$1

THIS_DIR=$(realpath $(dirname $0))

set -e
if [ ! -d $TARGET_DIR ]; then
  mkdir $TARGET_DIR
fi

# Download link for SemEval 2010 Task 8 data.
# This is a Google Drive link, but it seems to be the official one.
# For the website, see
# https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview
SEMEVAL_URL="https://drive.google.com/uc?authuser=0&id=0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk&export=download"

function fetch_data() {
  mkdir -p $TARGET_DIR/raw
  pushd $TARGET_DIR/raw

  # SemEval 2010 Task 8 official distribution (train and test).
  ZIPFILE_NAME="SemEval2010_task8_all_data.zip"
  wget "${SEMEVAL_URL}" -O "${ZIPFILE_NAME}"
  unzip "${ZIPFILE_NAME}"

  popd
}

fetch_data

# Convert SemEval to edge probing format.
TRAIN_SOURCE="$TARGET_DIR/raw/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
TEST_SOURCE="$TARGET_DIR/raw/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
python $THIS_DIR/convert-semeval2010-task8.py -i "${TRAIN_SOURCE}" \
    -o "$TARGET_DIR/train.all.json"
python $THIS_DIR/convert-semeval2010-task8.py -i "${TEST_SOURCE}" \
    -o "$TARGET_DIR/test.json"

# SemEval 2010 doesn't have an official development set,
# so create one by a (deterministic) random sample of the training data.
python ${THIS_DIR%jiant*}/jiant/probing/deterministic_split.py \
    -s 42 -f 0.85 -i "${TARGET_DIR}/train.all.json" \
    -o "${TARGET_DIR}/train.0.85.json" "${TARGET_DIR}/dev.json"

# Print dataset stats for sanity-check.
python ${THIS_DIR%jiant*}/jiant/probing/edge_data_stats.py -i $TARGET_DIR/*.json
