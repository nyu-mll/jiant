#!/bin/bash -l
#SBATCH --job-name=negation-eval-ccg
#SBATCH -o /home-3/nkim43@jhu.edu/log/naacl/probe/negation-eval-ccg.log
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpup100 --gres=gpu:1
#SBATCH -A t2-skhudan1
#SBATCH --cpus-per-task=6
#SBATCH –reservation=JSALT
#SBATCH --export=ALL
#SBATCH --mail-type=end
#SBATCH --mail-user=nkim43@jhu.edu
# run your job

module load python/3.6-anaconda
module load cuda
module load gcc/5.5.0
module load openmpi/3.1

source deactivate
conda deactivate
source activate jiant
source path_config_naacl.sh

export NFS_PROJECT_PREFIX=${PROBE_DIR}
export JIANT_PROJECT_PREFIX=${NFS_PROJECT_PREFIX}

PROBING_TASK=nli-prob-negation
EXP_NAME=negation
RUN_NAME=ccg

MODEL_DIR=${TRAIN_DIR}/nliprobing-mlp-train/ccg
#MODEL_DIR=/home-3/nkim43@jhu.edu/scratch/exp/naacl/train/ccg/mnli-train
PARAM_FILE=${MODEL_DIR}"/params.conf"
MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"
#MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_77.best_macro.th"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}"
OVERRIDES+=", use_classifier = mnli, is_probing_task=1, eval_data_fraction=1"
OVERRIDES+=", cuda = ${CUDA_VISIBLE_DEVICES}, load_model=1, do_target_task_training=0, reload_vocab=1"

python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf -o "${OVERRIDES}"

