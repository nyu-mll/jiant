# Preprocess, download, tokenization
# must point to transformers repo currently

export PYTHONPATH=jiant/

BASE_PATH=$(pwd)
JIANT_PATH=${BASE_PATH}/jiant/jiant
WORKING_DIR=${BASE_PATH}/experiments
DATA_DIR=${WORKING_DIR}/tasks
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache


function prepare_all_tasks() {
   MODEL_TYPE=$1
   TASKMASTER_TASKS=(boolq cb commonsenseqa copa rte wic snli qamr cosmosqa hellaswag wsc socialiqa arc_challenge arc_easy squad_v2 arct mnli piqa mutual mutual_plus quoref mrqa_natural_questions newsqa mcscript mctaco quail winogrande abductive_nli)
   for TASK_NAME in "${TASKMASTER_TASKS[@]}"
   do
       echo "run preprocess $TASK_NAME"
       preprocess_task $MODEL_TYPE $TASK_NAME
   done
}



function preprocess_task(){
    # Full model name e.g. nyu-mll/roberta-base-100M-1, roberta-base
    MODEL_TYPE=$1
    TASK_NAME=$2

    # Model name e.g. roberta-base-100M-1, roberta-base
    SHORT_MODEL_NAME="${MODEL_TYPE##*/}"
    echo "${SHORT_MODEL_NAME}: ${TASK_NAME}, ${DATA_DIR}"

    python ${JIANT_PATH}/proj/main/tokenize_and_cache.py \
        --task_config_path ${DATA_DIR}/configs/${TASK_NAME}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --phases train,val,test \
        --max_seq_length 256 \
        --do_iter \
        --smart_truncate \
        --output_dir ${CACHE_DIR}/${SHORT_MODEL_NAME}/${TASK_NAME}
}





