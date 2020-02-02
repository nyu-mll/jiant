#!/bin/sh
#$ -cwd
#$ -l short
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./cjlovering_config.sh; python main.py --config_file jiant/config/copa_bert.conf --overrides "exp_name = copa-apex-2, run_name = copa-2-gpu-apex-2-2-2-2"
