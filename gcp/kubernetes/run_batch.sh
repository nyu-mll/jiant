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

NAME=$1
COMMAND=$2

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
cat <<EOF | kubectl create -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: jiant-sandbox
        image: ${IMAGE}
        command: ["bash"]
        args: ["-l", "-c", "$COMMAND"]
        volumeMounts:
        - mountPath: /nfs/jsalt
          name: nfs-jsalt
        env:
        - name: NFS_PROJECT_PREFIX
          value: ${PROJECT_DIR}
        env:
        - name: JIANT_PROJECT_PREFIX
          value: ${PROJECT_DIR}
      volumes:
      - name: nfs-jsalt
        persistentVolumeClaim:
          claimName: nfs-jsalt-claim
          readOnly: false
EOF

