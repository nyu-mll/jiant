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
 
python main.py --config_file jiant/config/mtl.conf --overrides "exp_name=multitask_mlm, run_name=mlm_sst,do_pretrain=1, do_target_task_training=0, lr=5e-6, dropout=0.2, random_seed=922"
#python main.py --config_file jiant/config/taskmaster/base_roberta.conf --overrides "exp_name = task_lm, run_name = mlm_wikitext_seq512, target_tasks=mlm, do_pretrain=1, do_target_task_training=0, pretrain_tasks=mlm, accumulation_steps=2"
#python main.py --config_file jiant/config/taskmaster/base_roberta.conf --overrides "exp_name=mlm_wikitext, run_name=rte-superglue, target_tasks=rte-superglue, load_model=1, load_target_train_checkpoint=/misc/vlgscratch4/BowmanGroup/pmh330/jiant-outputs/datasize_control_v2/task_lm/mlm_wikitext_seq512/model_state_pretrain_val_59.best.th, pretrain_tasks=, input_module=roberta-large, batch_size=4, reload_vocab=1, do_pretrain=0, do_target_task_training=1, lr=5e-6, dropout=0.2, random_seed=922"
