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

set -eux

JIANT_PATH=$(readlink -f $(dirname $0))

TEMP_DIR=${1:-"/tmp/jiant-demo"}
mkdir -p $TEMP_DIR

# Set up temp dir to mimic our configuration of /nfs/jsalt
mkdir -p $TEMP_DIR/share
ln -sf $(readlink -f $(dirname $0)) $TEMP_DIR/share/jiant

# python scripts/download_glue_data.py --data_dir $TEMP_DIR/share/glue_data \
#   --tasks all

mkdir -p $TEMP_DIR/exp
chmod a+rwx $TEMP_DIR/exp

# Build the docker image. This will be slow the first time you run it,
# but will cache steps for subsequent runs.
IMAGE_NAME="jiant-sandbox:demo"
# sudo docker build -t $IMAGE_NAME .

# This is the python command we'll actually run;
# note that paths point to the fake nfs directory that the container will see.
declare -a COMMAND
COMMAND+=( python /share/jiant/main.py )
COMMAND+=( --config_file /share/jiant/config/demo.conf )
# COMMAND+=( ls -alh /share/jiant )

# Run demo.conf in the docker container.
sudo docker run --runtime=nvidia --rm -v "$TEMP_DIR:/nfs/jsalt" \
  -v "$JIANT_PATH:/share/jiant" \
  -e "NFS_PROJECT_PREFIX=/nfs/jsalt/exp" \
  -e "JIANT_PROJECT_PREFIX=/nfs/jsalt/exp" \
  -e "ELMO_SRC_DIR=" \
  -i ${IMAGE_NAME} "${COMMAND[@]}"
