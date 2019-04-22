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


#fine-tune on cola, then stop
python main.py --config_file config/spring19_seminar/cola_var/bert.conf --overrides "exp_name=test_cola_var_0422, run_name = run_01, bert_fine_tune = 1, do_pretrain = 1, pretrain_tasks = cola, do_target_task_training = 0, do_full_eval = 0, target_tasks = """

#loading a fine-tuned model on cola, test on questions
python main.py --config_file config/spring19_seminar/cola_var/bert_target_tasks.conf --overrides "exp_name=test_cola_var_0422, run_name = run_01, load_eval_checkpoint = none, target_tasks = npi_conditionals"

#target-task training for questions, without fine-tuning bert
python main.py --config_file config/spring19_seminar/cola_var/bert_target_tasks.conf --overrides "exp_name=test_cola_var_0422, run_name = run_02, allow_untrained_encoder_parameters = 1, load_eval_checkpoint = none, target_tasks = npi_conditionals"

