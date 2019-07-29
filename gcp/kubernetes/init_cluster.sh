#!/bin/bash

# Run this on a new Kubernetes cluster. See README.md for details.

pushd "$(dirname $0)"

kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml

YAML_STREAM=$( jsonnet -S templates/nfs.jsonnet )
echo "$YAML_STREAM"
echo "$YAML_STREAM" | kubectl apply -f -

