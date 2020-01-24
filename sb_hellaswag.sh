#!/bin/bash
#SBATCH --job-name=tense
#SBATCH --output=/scratch/pmh330/jiant-outputs/tense-%j.out
#SBATCH --error=/scratch/pmh330/jiant-outputs/tense-%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:p40:1
#SBATCH --exclude=gpu-36,gpu-37
#SBATCH --mem=100GB
#SBATCH --signal=USR1@600
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1


source activate jiant

#python main.py --config_file jiant/config/taskmaster/base_roberta.conf -o "exp_name=roberta-large, run_name=, target_tasks=, do_pretrain=1, do_target_task_training=0, input_module=roberta-large,pretrain_tasks=hellaswag,lr_patience=1,run_name=lrpatience1"


JIANT_CONF=$1
JIANT_OVERRIDES=$2
echo "$JIANT_CONF" 
echo "$JIANT_OVERRIDES"
#export JIANT_PROJECT_PREFIX="/scratch/pmh330/jiant-outputs/roberta-large-run2"
#JIANT_PROJECT_PREFIX="/scratch/pmh330/jiant-outputs/roberta-large-run2"
python main.py --config_file "$JIANT_CONF" -o "$JIANT_OVERRIDES" 
