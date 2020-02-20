#!/bin/bash
#SBATCH --job-name=mlm
#SBATCH --output=/misc/vlgscratch4/BowmanGroup/pmh330/jiant-outputs/tense-%j.out
#SBATCH --error=/misc/vlgscratch4/BowmanGroup/pmh330/jiant-outputs/tense-%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --signal=USR1@600
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1

module load anaconda3
source activate /misc/vlgscratch4/BowmanGroup/pmh330/conda/jiant
source user_config.sh
 
JIANT_CONF=$1
JIANT_OVERRIDES=$2
echo "$JIANT_CONF" 
echo "$JIANT_OVERRIDES"
python main.py --config_file "$JIANT_CONF" -o "$JIANT_OVERRIDES"
#python main.py --config_file jiant/config/mtl.conf --overrides "exp_name=multitask_mlm, run_name=mlm_sst$2,do_pretrain=1, do_target_task_training=0, lr=$1, dropout=0.2, random_seed=922"
