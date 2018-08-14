#!/bin/bash

# Helper script to update instance from image jsalt-image-1-3
# to equivalent of jsalt-image-1-4
set -e
set -x

# Copy configs to auto-mount NFS from new server.
# will mount to /nfs/jsalt
sudo cp -f $(dirname $0)/config/auto.master /etc
sudo cp $(dirname $0)/config/auto.nfs /etc
# Reload autofs daemon and check mount
sudo /etc/init.d/autofs restart
echo "Checking NFS mount at /nfs/jsalt. You should see files:"
ls -l /nfs/jsalt
echo ""

