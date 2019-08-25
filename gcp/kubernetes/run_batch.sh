#!/bin/bash

# Run a job on the Kubernetes cluster.
#
# Before running, be sure that:
#    - you're authenticated to the cluster via
#      gcloud container clusters get-credentials <cluster_name> --zone us-east1-c
#    - the resources defined in templates/jiant_env.libsonnet are correct
#
# Example usage:
# export JIANT_PATH="/nfs/jiant/home/$USER/jiant"
# ./run_batch.sh <job_name> "python $JIANT_PATH/main.py \
#    --config_file $JIANT_PATH/jiant/config/demo.conf \
#    --notify <your_email_address>"
#
# You can specify additional arguments as flags:
#    -m <mode>     # mode is 'create', 'replace', 'delete'
#    -g <gpu_type>  # e.g. 'k80' or 'p100'
#    -p <project>   # project folder to group experiments
#    -n <email>    # email address for job notifications
#
# For example:
# ./run_batch.sh -p demos -m k80 jiant-demo \
#     "python $JIANT_PATH/main.py --config_file $JIANT_PATH/jiant/config/demo.conf"
#
# will run as job name 'demos.jiant-demo' and write results to /nfs/jsalt/exp/demos
#
set -e

KUBECTL_MODE="create"
GPU_TYPE="p100"
PROJECT_NAME="$USER"
NOTIFY_EMAIL=""

# Get the NFS path from the Kubernetes config, so that it doesn't need to be
# hardcoded here.
pushd $(dirname $0)/templates
NFS_EXP_DIR=$(jsonnet -S -e "local env = import 'jiant_env.libsonnet'; env.nfs_exp_dir")
echo "Assuming NFS experiment path at $NFS_EXP_DIR"
popd

# Handle flags.
OPTIND=1         # Reset in case getopts has been used previously in the shell.
while getopts ":m:g:p:n:" opt; do
    case "$opt" in
    m)  KUBECTL_MODE=$OPTARG
        ;;
    g)  GPU_TYPE=$OPTARG
        ;;
    p)  PROJECT_NAME=$OPTARG
        ;;
    n)  NOTIFY_EMAIL=$OPTARG
        ;;
    \? )
        echo "Invalid flag $opt."
        exit 1
        ;;
    esac
done
shift $((OPTIND-1))

# Remaining positional args.
NAME=$1
COMMAND=$2

JOB_NAME="${USER}.${PROJECT_NAME}.${NAME}"

##
# Create project directory, if it doesn't exist yet.
PROJECT_DIR="${NFS_EXP_DIR}/${USER}/${PROJECT_NAME}"
if [ ! -d "${NFS_EXP_DIR}/$USER" ]; then
  mkdir "${NFS_EXP_DIR}/$USER"
fi
if [ ! -d "${PROJECT_DIR}" ]; then
  echo "Creating project directory ${PROJECT_DIR}"
  mkdir ${PROJECT_DIR}
  chmod -R o+w ${PROJECT_DIR}
fi

##
# Create custom config and save to project_dir.
YAML_DIR="${PROJECT_DIR}/yaml"
if [ ! -d "${YAML_DIR}" ]; then
  echo "Creating Kubernetes YAML ${YAML_DIR}"
  mkdir "${YAML_DIR}"
fi
# set -x  # uncomment for debugging
YAML_FILE="${PROJECT_DIR}/yaml/${JOB_NAME}.yaml"
jsonnet -S -o "${YAML_FILE}" \
  --tla-str job_name="${JOB_NAME}" \
  --tla-str command="${COMMAND}" \
  --tla-str project_dir="${PROJECT_DIR}" \
  --tla-str notify_email="${NOTIFY_EMAIL}" \
  --tla-str uid="${UID}" \
  --tla-str fsgroup="${GROUPS}" \
  --tla-str gpu_type="${GPU_TYPE}" \
  "$(dirname $0)/templates/run_batch.jsonnet"

##
# Create the Kubernetes pod; this will actually launch the job.
kubectl ${KUBECTL_MODE} -f "${YAML_FILE}"
