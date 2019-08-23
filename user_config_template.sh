#!/bin/bash

# Default environment variables for JSALT code. May be overwritten by user.
# See https://github.com/nyu-mll/jiant for more.

# DO NOT COMMIT CHANGES TO THIS FILE!
# Make a local copy and follow the instructions below.

# Copy this to /etc/profile.d/ to auto-set environment vars on login.
# Or, make a copy of this, customize, and run immediately before the training
# binary:
# cp user_config_template.sh user_config.sh
# source user_config.sh; python main.py --config jiant/config/demo.conf \
#   --overrides "do_pretrain = 0"
# ... or use scripts/export_from_bash.sh to set BASH up to run source automatically.

# (You can use a custom filename too, but user_config.sh will automatically be ignored by git.)

##
# Example of custom paths for a local installation:
# export JIANT_PROJECT_PREFIX=/Users/Bowman/Drive/JSALT
# export JIANT_DATA_DIR=/Users/Bowman/Drive/JSALT/jiant/glue_data

# The base directory for model output.
export JIANT_PROJECT_PREFIX=~

# Base directory in which to look for raw data subdirectories. This
# could be the glue_data directory created by download_glue_data.py.
export JIANT_DATA_DIR=~

# A word embeddings file in GloVe/fastText format. Not used when using
# ELMo, GPT, or BERT. To use more than one different set of embeddings
# in your environment, create an additional environment variable (like)
# FASTTEXT_WORD_EMBS_FILE, and reference it in each of your .conf config
# files with a line like:
#     word_embs_file = ${FASTTEXT_WORD_EMBS_FILE}
export WORD_EMBS_FILE=None

# Optional:
# echo "Loaded custom config."
