
# random-elmo
rm *.out
JIANT_OVERRIDES="exp_name = cola-elmo-baseline, run_name = random-elmo, allow_untrained_encoder_parameters = 1, allow_missing_task_map = 1, do_pretrain = 0" JIANT_CONF="config/cola-elmo.conf" sbatch ~/prince.sbatch

# elmo
JIANT_OVERRIDES="exp_name = cola-elmo-baseline, run_name = elmo, allow_reuse_of_pretraining_parameters = 1" JIANT_CONF="config/cola-elmo.conf" sbatch ~/prince.sbatch

JIANT_OVERRIDES="exp_name = cola-elmo-baseline, run_name = try, allow_reuse_of_pretraining_parameters = 1, do_target_tasks_training = 0" JIANT_CONF="config/cola-elmo.conf" sbatch ~/prince.sbatch


# elmo-ccg

# gpt
JIANT_OVERRIDES="exp_name = cola-gpt-baseline, run_name = gpt" JIANT_CONF="config/cola-openai.conf" sbatch ~/prince.sbatch

# gpt-ccg


srun --gres=gpu:k80:1 python main.py --config_file "config/cola-elmo.conf" --overrides "exp_name = cola-elmo-baseline, run_name = random-elmo, allow_untrained_encoder_parameters = 1, allow_missing_task_map = 1, do_pretrain = 0" 