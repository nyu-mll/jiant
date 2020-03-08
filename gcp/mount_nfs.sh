#!/bin/bash

# Mounts NFS at /nfs/jiant
# Ensure that config/auto.nfs has been updated with the correct IP Address and Fileshare name.

set -e
set -x

# Set up NFS automount
sudo apt-get -y install nfs-common autofs
sudo cp -f config/auto.master /etc
sudo cp -f config/auto.nfs /etc

# Reload autofs daemon and check mount
sudo /etc/init.d/autofs restart
echo "Checking NFS mount at /nfs/jiant. You should see files:"
ls -l /nfs/jiant
echo ""