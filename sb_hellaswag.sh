#!/bin/bash
#SBATCH --job-name=tense
#SBATCH --output=/scratch/pmh330/jiant-outputs/tense-%j.out
#SBATCH --error=/scratch/pmh330/jiant-outputs/tense-%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
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
python main.py --config_file "$JIANT_CONF" -o "$JIANT_OVERRIDES" 