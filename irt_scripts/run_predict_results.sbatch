#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --mail-type=END,FAIL

module purge
BASE_PATH=$(pwd)
JIANT_PATH=${BASE_PATH}/jiant/jiant
export PYTHONPATH=jiant/

model=$1
task=$2
config_no=$3
model_path=$4
mkdir -p ${BASE_PATH}/experiments/predict_files/${model}

echo "${model} ${task} ${config_no} ${model_path}"

SHORT_MODEL_NAME="${MODEL_TYPE##*/}"

python $JIANT_PATH/proj/main/runscript.py run   \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --model_path ${MODELS_DIR}/${SHORT_MODEL_NAME}/model/model.p \
    --model_config_path ${MODELS_DIR}/${SHORT_MODEL_NAME}/model/config.json \
    --jiant_task_container_config_path ${BASE_PATH}/experiments/run_config_dir/taskmaster/${model}/${task}_${config_no}/${task}.json  \
    --model_load_mode all --model_path ${BASE_PATH}/experiments/output_dir/taskmaster_${model}_bestconfig/${task}/config_${config_no}/${model_path}  \
    --output_dir  ${BASE_PATH}/experiments/predict_files/${model}/${task}_config_${config_no}_${model_path}  \
    --ZZoverrides model_path --write_test_preds 

