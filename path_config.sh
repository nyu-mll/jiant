#!/bin/bash

# Copy this to /etc/profile.d/ to auto-set environment vars on login.
# Or, customize this and run immediately before the training binary:
# source ../path_config.sh; python main.py --config ../config/demo.conf \
#   --overrides "do_train = 0"

# Default environment variables for JSALT code. May be overwritten by user.
# See https://github.com/jsalt18-sentence-repl/jiant for more.

export JSALT_SHARE_DIR="/media/jan/DATA/Projects/Research/Paul/DATA_FOR_JSALT"
export JIANT_DATA_DIR="$JSALT_SHARE_DIR/glue_data"

# Default experiment directory
export JIANT_PROJECT_PREFIX="$JSALT_SHARE_DIR/exp"

export GLOVE_EMBS_FILE="$JSALT_SHARE_DIR/glove/glove.840B.300d.txt"
export FASTTEXT_EMBS_FILE="$JSALT_SHARE_DIR/fasttext/crawl-300d-2M.vec"
export WORD_EMBS_FILE="$FASTTEXT_EMBS_FILE"
export FASTTEXT_MODEL_FILE="."  # not yet supported

export PATH_TO_COVE="$JSALT_SHARE_DIR/cove"

# pre-downloaded ELMo models
export ELMO_SRC_DIR="$JSALT_SHARE_DIR/elmo"

##
# Example of custom paths for a local installation:
# export JIANT_PROJECT_PREFIX=/Users/Bowman/Drive/JSALT
# export JIANT_DATA_DIR=/Users/Bowman/Drive/JSALT/jiant/glue_data
# export WORD_EMBS_FILE=~/glove.840B.300d.txt
# export FASTTEXT_MODEL_FILE=None
# export FASTTEXT_EMBS_FILE=None

# echo "Loaded Sam's config."

