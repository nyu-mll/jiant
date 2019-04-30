#!/bin/bash

# Quick-start: set up path_config.sh, run it, then run this
pushd "${PWD%jiant*}jiant"  # Make sure we're in the base jiant directory
python main.py --config_file config/demo.conf
