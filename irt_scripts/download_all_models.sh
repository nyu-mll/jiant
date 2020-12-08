export PYTHONPATH=jiant/

BASE_PATH=$(pwd)
JIANT_PATH=${BASE_PATH}/jiant/jiant
WORKING_DIR=${BASE_PATH}/experiments
DATA_DIR=${WORKING_DIR}/tasks
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache


function download_model(){
    MODEL_TYPE=$1
    if [[ $MODEL_TYPE = nyu* ]]
    then
        MODEL_NAME="${MODEL_TYPE##*/}"
        echo "Downloading miniBERTAs: $MODEL_NAME to ${MODELS_DIR}"
    	python ${JIANT_PATH}/proj/main/export_model.py \
            --model_type $MODEL_NAME \
            --hf_model_name $MODEL_TYPE \ 
	    --output_base_path ${MODELS_DIR}/${MODEL_NAME}
    else
	MODEL_NAME=${MODEL_TYPE}
        echo "Downloading ${MODEL_NAME} to ${MODELS_DIR}"
        python ${JIANT_PATH}/proj/main/export_model.py \
            --model_type ${MODEL_TYPE} \
            --output_base_path ${MODELS_DIR}/${MODEL_NAME}
    fi
}

function download_all_models(){
    MODEL_LIST=(albert-xxlarge-v2 roberta-base roberta-large nyu-mll/roberta-base-100M-1 nyu-mll/roberta-base-100M-2 nyu-mll/roberta-base-100M-3 nyu-mll/roberta-base-10M-1 nyu-mll/roberta-base-10M-2 nyu-mll/roberta-base-10M-3 nyu-mll/roberta-base-1B-1 nyu-mll/roberta-base-1B-2 nyu-mll/roberta-base-1B-3 nyu-mll/roberta-med-small-1M-1 nyu-mll/roberta-med-small-1M-2 nyu-mll/roberta-med-small-1M-3 bert-large-cased bert-base-cased xlm-roberta-large)

    for MODEL_TYPE in "${MODEL_LIST[@]}"
    do
        download_model $MODEL_TYPE	    
    done	    
}
