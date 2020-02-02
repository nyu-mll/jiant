#!/bin/sh
#$ -cwd
#$ -l short
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./cjlovering_config.sh; python main.py --config_file jiant/config/hans_bert.conf --overrides "exp_name = hans-bert-01, run_name = hans-bert-01"
