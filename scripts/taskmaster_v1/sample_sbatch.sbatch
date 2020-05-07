#!/bin/bash

# === Machine === #
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=16000
#SBATCH --time=24:00:00
#SBATCH --job-name="sample_sbatch_please_rename_this_thanks"


# === Environment === #
# 1. Do things like activate your environment, and enter the jiant folder, setup, etc
source scripts/taskmaster/transfer_analysis.sh

# === Run === #
# 2. Run things. Don't actually run this loop within the sbatch. Examples:
ez_first_intermediate_exp 1 sst
ez_run_intermediate_to_target_task 1 sst rte /path/to/proj
ez_run_intermediate_to_probing 1 sst edges-ner-ontonotes /path/to/proj
ez_run_intermediate_to_mixing 1 sst edges-ner-ontonotes /path/to/proj
