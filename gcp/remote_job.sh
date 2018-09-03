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

FULL_COMMAND="bash -l -c \"${COMMAND}\""

declare -a CL_ARGS

# Use internal ip between GCE instances.
# We use a DNS lookup to tell if the local host is a GCE instance. See
# https://stackoverflow.com/questions/30911775/how-to-know-if-a-machine-is-an-google-compute-engine-instance
dig_response=$(dig +short metadata.google.internal)
if [[ "$dig_response" != "" ]]; then
  CL_ARGS+=( --internal-ip )
fi
CL_ARGS+=( "${INSTANCE_NAME}" )
CL_ARGS+=( --command "${FULL_COMMAND}" )
CL_ARGS+=( --zone "$ZONE")
CL_ARGS+=( -- -t )

set -x
gcloud beta compute ssh "${CL_ARGS[@]}"

set +x
echo "Remote command completed successfully."

