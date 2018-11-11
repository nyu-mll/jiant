#!/bin/bash -l
#SBATCH --job-name=mlp-dissent
#SBATCH -o /home-3/nkim43@jhu.edu/log/naacl/train/mlp-dissent.log
#SBATCH --time=12:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpup100 --gres=gpu:1
#SBATCH --mem=15G
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

PROBING_TASK=mnli
EXP_NAME=nliprobing-mlp-train
RUN_NAME=dissent

MODEL_DIR=${TRAIN_DIR}/${RUN_NAME}/${RUN_NAME}-train
PARAM_FILE=${MODEL_DIR}"/params.conf"
#MODEL_FILE=${MODEL_DIR}"/model_state_eval_best.th"
MODEL_FILE=${MODEL_DIR}"/model_state_main_epoch_26.best_macro.th"

OVERRIDES="load_eval_checkpoint = ${MODEL_FILE}"
OVERRIDES+=", exp_name = ${EXP_NAME}"
OVERRIDES+=", run_name = ${RUN_NAME}"
OVERRIDES+=", target_tasks = ${PROBING_TASK}"
OVERRIDES+=", use_classifier = ${PROBING_TASK}, classifier=mlp, eval_data_fraction=1, is_probing_task=0"
OVERRIDES+=", cuda = ${CUDA_VISIBLE_DEVICES}, load_model=1, reload_vocab=0, elmo_chars_only=1, do_target_task_training=1"

python main.py -c config/final.conf ${PARAM_FILE} config/eval_existing.conf config/naacl_additional.conf -o "${OVERRIDES}" 

