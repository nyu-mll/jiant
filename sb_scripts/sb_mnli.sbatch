#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=jiant-mnli
#SBATCH --output=/scratch/pmh330/jiant-outputs/sb_jiant_mnli.out
#SBATCH --error=/scratch/pmh330/jiant-outputs/sb_jiant_mnli.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

source activate jiant
python main.py --config_file jiant/config/mnli_roberta.conf  >> /scratch/pmh330/jiant-outputs/out_train_mnli.log 2>&1
