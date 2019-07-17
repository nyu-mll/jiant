#!/bin/bash

# Set up a new workstation to use jiant
# Assumes that you've used the "Deep Learning Image: PyTorch 1.1.0" available on 
# Google Cloud Platform, and that you have an NFS volume mounted at /nfs/jiant.

set -e
set -x

pushd $(dirname $0)

# Set up NFS automount
sudo apt-get -y install nfs-common autofs
sudo cp -f config/auto.master /etc
sudo cp -f config/auto.nfs /etc

# Reload autofs daemon and check mount
sudo /etc/init.d/autofs restart
echo "Checking NFS mount at /nfs/jiant. You should see files:"
ls -l /nfs/jiant
echo ""

# Copy environment variables and set up paths
sudo cp -f config/jiant_paths.sh /etc/profile.d/jiant_paths.sh
source /etc/profile.d/jiant_paths.sh
if [ ! -d "${JIANT_PROJECT_PREFIX}" ]; then
  mkdir "${JIANT_PROJECT_PREFIX}"
fi
if [ ! -d "${PYTORCH_PRETRAINED_BERT_CACHE}" ]; then
  sudo mkdir -m 0777 "${PYTORCH_PRETRAINED_BERT_CACHE}"
fi

# Build the conda environment, and activate
pushd ..
conda env create -f environment.yml
conda activate jiant

# Download NLTK packages
python -c "import nltk; nltk.download('perluniprops')"
python -c "import nltk; nltk.download('nonbreaking_prefixes')"

echo "Set-up complete! You may need to run 'source /etc/profile.d/jiant_paths.sh', or log out and log back in for things to work."

