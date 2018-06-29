#!/bin/bash

# Helper script to update instance from image jsalt-image-1-0
# to equivalent of jsalt-image-1-1
set -e
set -x

# Install google cloud logging
sudo /usr/share/anaconda3/bin/pip install --upgrade google-cloud-logging

# Copy ELMo data from cloud bucket
gsutil cp -R gs://jsalt-data/elmo /tmp
sudo rm -r /usr/share/jsalt/elmo
sudo mv -f /tmp/elmo /usr/share/jsalt
# Should see two files here
ls -l /usr/share/jsalt/elmo

# Copy updated paths file to /etc/profile.d
sudo cp $(dirname $0)/config/jsalt_paths.sh /etc/profile.d/

echo 'Updated! At next login, you should have $ELMO_SRC_DIR set to point to downloaded ELMo models.'
