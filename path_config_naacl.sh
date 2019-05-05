#!/bin/bash

# DO NOT COMMIT CHANGES TO THIS FILE! Make a local copy and follow the
# instructions below.

# Copy this to /etc/profile.d/ to auto-set environment vars on login.
# Or, make a copy of this, customize, and run immediately before the training
# binary:
# cp path_config.sh ~/my_path_config.sh
# source ~/my_path_config.sh; python main.py --config ../config/demo.conf \
#   --overrides "do_pretrain = 0"

# Default environment variables for JSALT code. May be overwritten by user.
# See https://github.com/jsalt18-sentence-repl/jiant for more.

##
# Example of custom paths for a local installation:
# export JIANT_PROJECT_PREFIX=/Users/Bowman/Drive/JSALT
# export JIANT_DATA_DIR=/Users/Bowman/Drive/JSALT/jiant/glue_data
# export JIANT_DATA_DIR=/home/raghu1991_p_gmail_com/
# export WORD_EMBS_FILE=~/glove.840B.300d.txt
# export FASTTEXT_MODEL_FILE=None
# export FASTTEXT_EMBS_FILE=None

export PREFIX=/export/fs01/nk/exp/naacl
export PROBE_PREFIX=/export/fs01/nk/exp/starsem
export NFS_DATA_DIR=/export/a12/nk/share/glue_data
export JIANT_DATA_DIR=${NFS_DATA_DIR}
export WORD_EMBS_FILE=~/data/glove/glove.840B.300d.txt
export FASTTEXT_MODEL_FILE=None
export FASTTEXT_EMBS_FILE=None
export PATH_TO_COVE=/export/a12/nk/share/cove

export TRAIN_DIR=${PREFIX}/train
export PROBE_DIR=${PROBE_PREFIX}/probe

export PRETRAIN_MNLI=${TRAIN_DIR}/mnli
export PRETRAIN_CCG=${TRAIN_DIR}/ccg

echo "Loaded path_config_naacl."

