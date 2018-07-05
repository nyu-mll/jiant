#!/bin/bash

# Convenience script to copy this jiant repo to a (new) GCE instance.
# Usage: ./transfer_code.sh <instance_name> [<zone>]
echo "FRIENDLY WARNING: Your life will be easier if you keep your code of NFS and maintain only one copy."

INSTANCE_NAME="${1:-"$USER"}"
ZONE="${2:-"us-east1-c"}"

set -e

if [ -z $INSTANCE_NAME ]; then
  echo "You must provide an instance name!"
  exit 1
fi

# Get parent dir of this script's dir
JIANT_DIR=$(cd `dirname $0`/../; pwd)

set -x
gcloud compute scp --project jsalt-sentence-rep --zone "$ZONE" \
  --recurse $JIANT_DIR "$INSTANCE_NAME:~"

echo "Transfer completed successfully."
