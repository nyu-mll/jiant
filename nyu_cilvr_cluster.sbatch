#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -t24:00:00
#SBATCH --constraint=gpu_12gb,pascal

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $JIANT_FLAGS >> ~/jiant_machine_assignments.txt

# Run.
conda activate jiant
cd ~/jiant/src
source ../user_config.sh
python main.py --config_file ../config/defaults.conf --overrides "global_ro_exp_dir = /misc/vlgscratch4/BowmanGroup/sbowman/exp/default, project_dir = $JIANT_PROJECT_PREFIX, $JIANT_OVERRIDES"
