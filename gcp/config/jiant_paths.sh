#!/bin/bash

# Copy this to /etc/profile.d/ to auto-set environment vars on login.

# Default environment variables for jiant. May be overwritten by user.
# See https://github.com/nyu-mll/jiant for more.

# Default location for glue_data
export JIANT_DATA_DIR="/nfs/jiant/share/glue_data"
# Default experiment parent directory
export JIANT_PROJECT_PREFIX="$HOME/exp"

# pre-downloaded ELMo models
export ELMO_SRC_DIR="/nfs/jiant/share/elmo"
# cache for BERT etc. models
export HUGGINGFACE_TRANSFORMERS_CACHE="/nfs/jiant/share/transformers_cache"
export PYTORCH_TRANSFORMERS_CACHE="/nfs/jiant/share/transformers_cache"
# word embeddings
export WORD_EMBS_FILE="/nfs/jiant/share/wiki-news-300d-1M.vec"

