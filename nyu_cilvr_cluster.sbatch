#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.

#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --constraint=gpu_12gb,pascal

# Example usage:
# JIANT_OVERRIDES="exp_name = main-multi-task-glue, train_tasks = glue, run_name = noelmo-do2-sd1, elmo_chars_only = 1, dropout = 0.2" sbatch ../nyu_cilvr_cluster.sbatch

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $JIANT_OVERRIDES >> ~/jiant_machine_assignments.txt

# Run.
cd ~/jiant/src
source ../user_config.sh
python ../main.py --config_file ../config/defaults.conf --overrides "global_ro_exp_dir = /misc/vlgscratch4/BowmanGroup/sbowman/exp/default, project_dir = $JIANT_PROJECT_PREFIX, $JIANT_OVERRIDES"
