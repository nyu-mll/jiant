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

set -x
gcloud compute ssh --zone="$ZONE" "${INSTANCE_NAME}" \
  --command="${FULL_COMMAND}" -- -t


