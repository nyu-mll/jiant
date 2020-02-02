#!/bin/sh
#$ -cwd
#$ -l short
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./cjlovering_config.sh; python main.py --config_file jiant/config/mnli_bert.conf --overrides "exp_name = bert-mnli-01, run_name = bert-mnli-01"
