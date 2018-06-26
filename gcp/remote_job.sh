#!/bin/bash

# Convenience script to execute a command (such as a training script) remotely 
# on a worker instance. 

# This is the simple version, which runs "synchronously": 
# the current shell controls the process, and hitting Ctrl+C will kill the 
# command (as will losing the SSH connection). 
# For persistence, use remote_job_tmux.sh

# Usage: ./remote_job.sh <instance_name> <command> [<zone>]
INSTANCE_NAME="${1}"
COMMAND="${2:-"nvidia-smi"}"
ZONE="${3:-"us-east1-c"}"

set -e

if [ -z $INSTANCE_NAME ]; then
  echo "You must provide an instance name!"
  exit 1
fi

FULL_COMMAND="bash -l -c ${COMMAND}"

set -x
gcloud compute ssh --project jsalt-sentence-rep --zone "$ZONE" \
  "${INSTANCE_NAME}" --command="${FULL_COMMAND}"

set +x
echo "Remote command completed successfully."

