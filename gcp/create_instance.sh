#!/bin/bash

# Helper script to create a GCP instance from one of the default templates

# Usage: ./create_instance.sh <instance_type> <instance_name> [<zone>]
# Instance type is one of:
#   workstation :  CPU workstation
#   gpu-1       :  single-GPU worker
#   gpu-2       :  double-GPU worker
#   gpu-4       :  quad-GPU worker
INSTANCE_TYPE="${1:-'gpu-1'}"
INSTANCE_NAME="${2}"
ZONE="${3:-"us-east1-c"}"

set -e

if [ -z $INSTANCE_NAME ]; then
  echo "You must provide an instance name!"
  exit 1
fi

if [[ $INSTANCE_TYPE == "cpu" || $INSTANCE_TYPE == "workstation" ]]; then
  TEMPLATE="cpu-workstation-template"
elif [[ $INSTANCE_TYPE == "gpu-1" ]]; then
  TEMPLATE="gpu-worker-template-1"
elif [[ $INSTANCE_TYPE == "gpu-2" ]]; then
  TEMPLATE="gpu-worker-template-2"
elif [[ $INSTANCE_TYPE == "gpu-4" ]]; then
  TEMPLATE="gpu-worker-template-4"
else
  echo "Unsupported instance type '$INSTANCE_TYPE'"
  exit 1
fi

set -x
gcloud compute instances create "$INSTANCE_NAME" \
  --project jsalt-sentence-rep --zone "$ZONE" \
  --source-instance-template "$TEMPLATE"

echo "Instance created! Wait a minute or two before attempting to SSH."
STATUS_URL="https://console.cloud.google.com/compute/instancesDetail/zones/$ZONE/instances/$INSTANCE_NAME?project=jsalt-sentence-rep"
echo "You can monitor status at: $STATUS_URL"
