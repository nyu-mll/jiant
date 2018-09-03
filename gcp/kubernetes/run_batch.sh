#!/bin/bash

# Run a batch job on the Kubernetes cluster.
#
# Before running, be sure that:
#    - the image is built and available at $IMAGE, below.
#    - you're authenticated to the cluster via
#      gcloud container clusters get-credentials <cluster_name> --zone us-east1-c
#
# Example usage:
# export JIANT_PATH="/nfs/jsalt/home/$USER/jiant"
# ./run_batch.sh <job_name> "python $JIANT_PATH/main.py \
#    --config_file $JIANT_PATH/config/demo.conf \
#    --notify <your_email_address>"
#
# The third argument, MODE, can be used to 'replace' an existing job (e.g. if
# you mis-configured and it failed on startup), or to 'delete' one from the
# cluster.

NAME=$1
COMMAND=$2
MODE=${3:-"create"}    # create, replace, delete
GPU_TYPE=${4:-"p100"}  # k80 or p100

JOB_NAME="${USER}.${NAME}"
PROJECT_DIR="/nfs/jsalt/exp/$USER"
if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Creating project directory ${PROJECT_DIR}"
  mkdir ${PROJECT_DIR}
fi

GCP_PROJECT_ID="$(gcloud config get-value project -q)"
IMAGE="gcr.io/${GCP_PROJECT_ID}/jiant-sandbox:v1"

##
# Create custom config and create a Kubernetes job.
cat <<EOF | kubectl ${MODE} -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: jiant-sandbox
        image: ${IMAGE}
        command: ["bash"]
        args: ["-l", "-c", "$COMMAND"]
        resources:
          limits:
           nvidia.com/gpu: 1
        volumeMounts:
        - mountPath: /nfs/jsalt
          name: nfs-jsalt
        env:
        - name: NFS_PROJECT_PREFIX
          value: ${PROJECT_DIR}
        - name: JIANT_PROJECT_PREFIX
          value: ${PROJECT_DIR}
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-${GPU_TYPE}
      volumes:
      - name: nfs-jsalt
        persistentVolumeClaim:
          claimName: nfs-jsalt-claim
          readOnly: false
EOF

