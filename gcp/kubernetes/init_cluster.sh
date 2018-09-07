#!/bin/bash

# Run this on a new Kubernetes cluster. See README.md for details.

pushd "${PWD%jiant*}"/jiant

kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml

kubectl create -f gcp/kubernetes/nfs_pv.yaml
kubectl create -f gcp/kubernetes/nfs_pvc.yaml


