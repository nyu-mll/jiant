#!/bin/bash

# Convenience script to execute a command (such as a training script) remotely
# on a worker instance.

# This version creates a tmux session on the remote machine and runs the command
# in it. This gives persistence: if you disconnect from the remote machine, the
# job will continue running.
#
# To exit the tmux session on the remote host, press Ctrl+b followed by the 'd'
# key. To re-attach, you can ssh to the remote host and run:
#   tmux attach -t <job_name>
# Or run this script with the command 'A'
#
# To kill a target job and its associated tmux session, run with the command 'K'.

# Usage: ./remote_job_tmux.sh <instance_name> <job_name> <command> [<zone>]
INSTANCE_NAME="${1}"
JOB_NAME="${2}"
COMMAND="${3:-"a"}"
ZONE="${4:-"us-east1-c"}"

set -e

if [ -z $INSTANCE_NAME ]; then
  echo "You must provide an instance name!"
  exit 1
fi
if [ -z $JOB_NAME ]; then
  echo "You must provide a job name!"
  exit 1
fi

if [[ $COMMAND == "A" || $COMMAND == "a" ]]; then
  FULL_COMMAND="tmux attach -t ${JOB_NAME}"
elif [[ $COMMAND == "K" || $COMMAND == "k" ]]; then
  FULL_COMMAND="tmux kill-session -t ${JOB_NAME}"
else
  FULL_COMMAND="tmux new -s ${JOB_NAME} -d; "
  FULL_COMMAND+="tmux send '$COMMAND' Enter; "
  FULL_COMMAND+="tmux attach -t ${JOB_NAME}"
fi

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


