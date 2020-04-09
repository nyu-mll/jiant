#!/bin/bash
#SBATCH --job-name=K_mlm
#SBATCH --output=/scratch/pmh330/jiant-outputs/tense-%j.out
#SBATCH --error=/scratch/pmh330/jiant-outputs/tense-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:2 
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --signal=USR1@600
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL
module load anaconda3/5.3.1
source activate jiant
source user_config.sh

JIANT_CONF=$1
JIANT_OVERRIDES=$2
echo "$JIANT_CONF" 
echo "$JIANT_OVERRIDES"
~/miniconda3/envs/jiant/bin/python main.py --config_file "$JIANT_CONF" -o "$JIANT_OVERRIDES"

