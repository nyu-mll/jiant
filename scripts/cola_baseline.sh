# baseline experiments on cola.
# don't actually run this sh file as a whole,
# copy the lines you need and paste it to the command line.
# yes, this sounds stupid, I know.

# set up environment
module purge
module load tensorflow/python3.6/1.5.0
module swap python3/intel anaconda3/5.3.1
source activate jiant
export NFS_PROJECT_PREFIX=/scratch/hl3236/ling3340
export JIANT_DATA_DIR=/home/hl3236/data
export WORD_EMBS_FILE=/home/hl3236/misc/crawl-200d-2M.vec

# random-elmo
srun --gres=gpu:k80:1 python main.py --config_file "config/cola-elmo.conf" --overrides "exp_name = cola-elmo-baseline, run_name = random-elmo, allow_untrained_encoder_parameters = 1, allow_missing_task_map = 1, do_pretrain = 0" 
# JIANT_OVERRIDES="exp_name = cola-elmo-baseline, run_name = random-elmo, allow_untrained_encoder_parameters = 1, allow_missing_task_map = 1, do_pretrain = 0" JIANT_CONF="config/cola-elmo.conf" sbatch ~/prince.sbatch

# elmo
srun --gres=gpu:k80:1 python main.py --config_file "config/cola-elmo.conf" --overrides "exp_name = cola-elmo-baseline, run_name = elmo" 
# JIANT_OVERRIDES="exp_name = cola-elmo-baseline, run_name = elmo JIANT_CONF="config/cola-elmo.conf" sbatch nyu_cilvr_cluster.sbatch

# gpt
srun --gres=gpu:k80:1 python main.py --config_file "cconfig/cola-openai.conf" --overrides "exp_name = cola-gpt-baseline, run_name = gpt" 
# JIANT_OVERRIDES="exp_name = cola-gpt-baseline, run_name = gpt" JIANT_CONF="config/cola-openai.conf" sbatch nyu_cilvr_cluster.sbatch
