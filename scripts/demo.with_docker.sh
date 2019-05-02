#!/bin/bash

# Run jiant with demo.conf in a docker container. 
#
# The container expects a number of paths relative to /nfs/jsalt;
# for now, we'll fake these by setting up a temp directory.
#
# TODO to replace this with a properly-agnostic mount point inside the
# container.
#
# TODO: turn this into a real integration test...

# temporarily cd to jiant/
pushd $(dirname $0)
pushd $(git rev-parse --show-toplevel)

set -eux

# Get the path to this repo; we'll mount it from the container later.
JIANT_PATH=$(readlink -f .)

TEMP_DIR=${1:-"/tmp/jiant-demo"}

# Set up temp dir to mimic our configuration of /nfs/jsalt
mkdir -p $TEMP_DIR
mkdir -p $TEMP_DIR/exp
mkdir -p $TEMP_DIR/share
mkdir -p $TEMP_DIR/share/bert_cache
python scripts/download_glue_data.py --data_dir $TEMP_DIR/share/glue_data \
  --tasks all

# Build the docker image. This will be slow the first time you run it,
# but will cache steps for subsequent runs.
IMAGE_NAME="jiant-sandbox:demo"
sudo docker build -t $IMAGE_NAME .

# This is the python command we'll actually run, with paths relative to the
# container root. See the -v "src:dst" flags below for the mapping.
declare -a COMMAND
COMMAND+=( python /share/jiant/main.py )
COMMAND+=( --config_file /share/jiant/config/demo.conf )
COMMAND+=( -o "exp_name=jiant-demo" )

# Run demo.conf in the docker container.
sudo docker run --runtime=nvidia --rm -v "$TEMP_DIR:/nfs/jsalt" \
  -v "$JIANT_PATH:/share/jiant" \
  -e "JIANT_PROJECT_PREFIX=/nfs/jsalt/exp" \
  -e "PYTORCH_PRETRAINED_BERT_CACHE=/nfs/jsalt/share/bert_cache" \
  -e "ELMO_SRC_DIR=" \
  --user $(id -u):$(id -g) \
  -i ${IMAGE_NAME} "${COMMAND[@]}"
