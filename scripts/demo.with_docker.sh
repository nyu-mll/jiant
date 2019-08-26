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

NFS_PATH="/nfs/jiant"  # absolute path to NFS volume
mkdir -p $NFS_PATH/exp/$USER  # project directory

# Build the docker image. This will be slow the first time you run it,
# but will cache steps for subsequent runs.
IMAGE_NAME="jiant-sandbox:demo"
sudo docker build --build-arg "NFS_MOUNT=$NFS_PATH" -t $IMAGE_NAME .

# This is the python command we'll actually run, with paths relative to the
# container root. See the -v "src:dst" flags below for the mapping.
declare -a COMMAND
COMMAND+=( python $NFS_PATH/jiant/main.py )
COMMAND+=( --config_file $NFS_PATH/jiant/jiant/config/demo.conf )
COMMAND+=( -o "exp_name=jiant-demo" )

# Run demo.conf in the docker container.
sudo docker run --runtime=nvidia --rm \
  -v "$NFS_PATH:$NFS_PATH" \
  -v "$JIANT_PATH:$NFS_PATH/jiant" \
  -e "JIANT_DATA_DIR=$JIANT_DATA_DIR" \
  -e "ELMO_SRC_DIR=$ELMO_SRC_DIR" \
  -e "WORD_EMBS_FILE=$WORD_EMBS_FILE" \
  -e "JIANT_PROJECT_PREFIX=$NFS_PATH/exp/$USER" \
  -e "PYTORCH_PRETRAINED_BERT_CACHE=$PYTORCH_PRETRAINED_BERT_CACHE" \
  --user $(id -u):$(id -g) \
  -i ${IMAGE_NAME} "${COMMAND[@]}"
