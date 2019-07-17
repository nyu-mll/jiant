#!/bin/bash

# Set up a new workstation to use jiant
# Assumes that you've used the "Deep Learning Image: PyTorch 1.1.0" available on 
# Google Cloud Platform, and that you have an NFS volume mounted at /nfs/jiant.

set -e
set -x

# Set up NFS automount
sudo apt-get -y install nfs-common autofs
sudo cp -f $(dirname $0)/config/auto.master /etc
sudo cp -f $(dirname $0)/config/auto.nfs /etc

# Reload autofs daemon and check mount
sudo /etc/init.d/autofs restart
echo "Checking NFS mount at /nfs/jiant. You should see files:"
ls -l /nfs/jiant
echo ""

# Copy environment variables and set up paths
sudo cp -f $(dirname $0)/config/jiant_paths.sh /etc/profile.d/jiant_paths.sh
source /etc/profile.d/jiant_paths.sh
if [ ! -d "${JIANT_PROJECT_PREFIX}" ]; then
  sudo mkdir "${JIANT_PROJECT_PREFIX}"
fi
if [ ! -d "${PYTORCH_PRETRAINED_BERT_CACHE}" ]; then
  sudo mkdir "${PYTORCH_PRETRAINED_BERT_CACHE}"
fi

# Install packages
sudo $(which pip) install --upgrade google-cloud-logging
sudo $(which pip) install sendgrid
sudo $(which pip) install python-Levenshtein


