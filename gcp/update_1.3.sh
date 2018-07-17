#!/bin/bash

# Helper script to update instance from image jsalt-image-1-2
# to equivalent of jsalt-image-1-3
set -e
set -x

# Install Python packages
sudo /usr/share/anaconda3/bin/pip install sendgrid
sudo /usr/share/anaconda3/bin/pip install python-Levenshtein

set +x
echo 'Updated!'

