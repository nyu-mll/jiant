#!/bin/sh
#$ -cwd
#$ -l short
#$ -l gpus=2
#$ -e ./logs/
#$ -o ./logs/

mkdir -p ./logs/
. ~/.bashrc
conda activate jiant
. ./cjlovering_config.sh; python main.py --config_file jiant/config/task_1.conf --overrides "exp_name = task-1-01, run_name = task-1-01"
