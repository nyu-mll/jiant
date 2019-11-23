#!/bin/bash
#SBATCH --job-name=squad
#SBATCH --output=/scratch/pmh330/jiant-outputs/squad-%j.out
#SBATCH --error=/scratch/pmh330/jiant-outputs/squad-%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=80GB
#SBATCH --signal=USR1@600
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1


source activate jiant

python main.py --config_file jiant/config/taskmaster/base_roberta.conf -o "exp_name=roberta-large, target_tasks=, do_pretrain=1, do_target_task_training=0, input_module=roberta-large,pretrain_tasks=squad,lr_patience=1,run_name=squad"

