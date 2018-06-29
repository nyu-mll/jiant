#!/bin/bash

# Helper script to update instance from image jsalt-image-1-1
# to equivalent of jsalt-image-1-2
set -e
set -x

# Install NFS
sudo apt-get -y update
sudo apt-get -y install nfs-common autofs

# Copy configs to auto-mount NFS
# will mount to /nfs/jsalt
sudo cp -f $(dirname $0)/config/auto.master /etc
sudo cp $(dirname $0)/config/auto.nfs /etc
# Reload autofs daemon and check mount
sudo /etc/init.d/autofs restart
echo "Checking NFS mount at /nfs/jsalt. You should see files:"
ls -l /nfs/jsalt
echo ""

# Copy updated paths file to /etc/profile.d
sudo cp -f $(dirname $0)/config/jsalt_paths.1.2.sh /etc/profile.d/jsalt_paths.sh

set +x
echo 'Updated! Be sure to re-start shells or type:'
echo '  source /etc/profile.d/jsalt_paths.sh'
echo 'to set updated environment variables in each shell you have open.'
echo '(if this is confusing, just re-start your instance...)'

