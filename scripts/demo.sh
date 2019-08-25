#!/bin/bash

# Quick-start: use path_config.sh to set up your environment variables, then run this
pushd "${PWD%jiant*}jiant"  # Make sure we're in the base jiant directory
python main.py --config_file jiant/config/demo.conf
