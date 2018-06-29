#!/bin/bash

# Copy this to /etc/profile.d/ to auto-set environment vars on login.

# Default environment variables for JSALT code. May be overwritten by user.
# See https://github.com/jsalt18-sentence-repl/jiant for more.

# export JSALT_SHARE_DIR="/usr/share/jsalt"
export JSALT_SHARE_DIR="/nfs/jsalt/share"
export JIANT_DATA_DIR="$JSALT_SHARE_DIR/glue_data"

# Default experiment directory
export JIANT_PROJECT_PREFIX="$HOME/exp"
export NFS_PROJECT_PREFIX="/nfs/jsalt/exp/$HOSTNAME"

export GLOVE_EMBS_FILE="$JSALT_SHARE_DIR/glove/glove.840B.300d.txt"
export FASTTEXT_EMBS_FILE="$JSALT_SHARE_DIR/fasttext/crawl-300d-2M.vec"
export WORD_EMBS_FILE="$FASTTEXT_EMBS_FILE"
export FASTTEXT_MODEL_FILE="."  # not yet supported

export PATH_TO_COVE="$JSALT_SHARE_DIR/cove"

# pre-downloaded ELMo models
export ELMO_SRC_DIR="$JSALT_SHARE_DIR/elmo"

