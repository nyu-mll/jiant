#!/bin/bash -l
#SBATCH --job-name=mnli-train
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /home-3/nkim43@jhu.edu/log/naacl/train/mnli-train.log
#SBATCH -p gpup100 --gres=gpu:1
#SBATCH -A t2-skhudan1
#SBATCH --cpus-per-task=6
#SBATCH –reservation=JSALT
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=nkim43@jhu.edu

module load python/3.6-anaconda
module load cuda
module load gcc/5.5.0
module load openmpi/3.1

source deactivate
conda deactivate
source activate jiant
source path_config_naacl.sh

export NFS_PROJECT_PREFIX=${TRAIN_DIR}
export JIANT_PROJECT_PREFIX=${NFS_PROJECT_PREFIX}

EXP_NAME=mnli
RUN_NAME=mnli-train

OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", elmo_chars_only=1, dropout=0.2, cuda=${CUDA_VISIBLE_DEVICES}"
OVERRIDES+=', pretrain_tasks="mnli", target_tasks=none'
OVERRIDES+=", reload_vocab=0, reload_tasks=0, training_data_fraction=1, do_target_task_training=0, do_full_eval=0"
OVERRIDES+=", load_model=1"

python main.py --config_file config/final.conf config/naacl_additional.conf -o "${OVERRIDES}"

