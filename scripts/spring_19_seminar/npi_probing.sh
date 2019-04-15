#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --job-name=npi_probing
#SBATCH --mail-type=END
##SBATCH --mail-user=sfw268@nyu.edu
#SBARCH --mem=320GB
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1



python main.py --config_file config/npi_probing_finetune.conf --overrides "exp_name=npi_probing_test, run_name = run_01, bert_fine_tune = 1, do_pretrain = 1, pretrain_tasks = npi, do_target_task_training = 0, do_full_eval = 0"

python main.py --config_file config/npi_probing_target_task.conf --overrides "exp_name=npi_probing_test, run_name = run_01"

