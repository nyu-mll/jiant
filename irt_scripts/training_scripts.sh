BASE_DIR=$(pwd)
WORKING_DIR=${BASE_DIR}/experiments
DATA_DIR=${WORKING_DIR}/tasks
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache
RUN_CONFIG_DIR=${WORKING_DIR}/run_config_dir/taskmaster
OUTPUT_DIR=${WORKING_DIR}/output_dir/taskmaster

export PYTHONPATH=jiant/

declare -A TRAIN_SIZE=(
  ["sst"]=67000
  ["mnli"]=392703
  ["adversarial_nli_r1"]=330000
  ["scitail"]=23596
  ["qasrl"]=250000
  ["qamr"]=62270
  ["squad_v2"]=130319
  ["cosmosqa"]=25263
  ["hellaswag"]=39905
  ["commonsenseqa"]=9741
  ["abductive_nli"]=169654
  ["arc_easy"]=2251
  ["arc_challenge"]=1119
  ["newsqa"]=11469
  ["mrqa_natural_questions"]=104071
  ["mcscript"]=14191
  ["mutual"]=7088
  ["mutual_plus"]=7088
  ["mctaco"]=3026
  ["mctest160"]=1200
  ["mctest500"]=280
  ["piqa"]=16113
  ["quoref"]=19399
  ["winogrande"]=40398
  ["quail"]=10246
  ["rte"]=2491
  ["socialiqa"]=33409
  ["boolq"]=9427
  ["cb"]=250
  ["copa"]=400
  ["multirc"]=5100
  ["arct"]=1210
  ["record"]=65709
  ["wic"]=5428
  ["snli"]=570000
  ["wsc"]=554
  ["stsb"]=7000
  ["qqp"]=364000
  ["wnli"]=634
)

BIG_TASKS=(adversarial_nli_r1 mnli mnli_mismatched squad_v2 record cosmosqa hellaswag abductive_nli arc_easy arc_challenge quoref newsqa mrqa_natural_questions mcscript mutual_plus mutual piqa quail)


function run_all_configs() {
    MODEL_TYPE=$1
    TASK_NAME=$2
    echo "$OUTPUT_DIR"

    echo "train size: ${TRAIN_SIZE[${TASK_NAME}]}"
    if ((  ${TRAIN_SIZE[${TASK_NAME}]} < 5000 )); then
        # use epoch 10 and 40
        CONFIG_NUMS=( 1 2 4 5 7 8 )
    else
        # use epoch 3 and 10
        CONFIG_NUMS=( 0 1 3 4 6 7 ) 
    fi


    for CONFIG_NO in ${CONFIG_NUMS[@]}
    do
        run_training ${MODEL_TYPE} ${TASK_NAME} ${CONFIG_NO}
    done
}


function run_training() {
    MODEL_TYPE=$1
    TASK_NAME=$2
    CONFIG_NO=$3
    SEED=1111


    if [[ " ${BIG_TASKS[*]} " == *" ${TASK_NAME} "* ]]; then
        train_batch_size=4
        grad_acc=4
    else
        train_batch_size=""
        grad_acc=1
    fi

    if [[ $train_batch_size == "" ]]
    then
        if [[ $MODEL_TYPE = albert* ]]
        then
            train_batch_size=8
            grad_acc=2
            echo "albert ${grad_acc}"
        else
            train_batch_size=16
            grad_acc=1
        fi
    fi

    echo "${MODEL_TYPE} bsz ${train_batch_size} grad ${grad_acc} $TASK_NAME"

    ###
    VAL_TASKS=$TASK_NAME
    TRAIN_TASKS=$TASK_NAME
    if [[ ${TASK_NAME} == "mnli" ]]; then
        VAL_TASKS="mnli,mnli_mismatched"
    elif [[ ${TASK_NAME} == "adversarial_nli_r1" ]]; then
        TRAIN_TASKS="adversarial_nli_r1,adversarial_nli_r2,adversarial_nli_r3,mnli,snli"
        VAL_TASKS="adversarial_nli_r1,adversarial_nli_r2,adversarial_nli_r3,mnli,mnli_mismatched,snli"
    fi
    ###

    val_interval=$(expr ${TRAIN_SIZE[${TASK_NAME}]} / $train_batch_size)
    val_interval=$((val_interval>5000 ? 5000 : val_interval))

    if [[ ${CONFIG_NO} == 0 ]]; then
        lr=1e-5
        epochs=3
    elif [[ ${CONFIG_NO} == 1 ]]; then
        lr=1e-5
        epochs=10
    elif [[ ${CONFIG_NO} == 2 ]]; then
        lr=1e-5
        epochs=40
    elif [[ ${CONFIG_NO} == 3 ]]; then
        lr=3e-5
        epochs=3
    elif [[ ${CONFIG_NO} == 4 ]]; then
        lr=3e-5
        epochs=10
    elif [[ ${CONFIG_NO} == 5 ]]; then
        lr=3e-5
        epochs=40
    elif [[ ${CONFIG_NO} == 6 ]]; then
        lr=5e-6
        epochs=3
    elif [[ ${CONFIG_NO} == 7 ]]; then
        lr=5e-6
        epochs=10
    elif [[ ${CONFIG_NO} == 8 ]]; then
        lr=5e-6
        epochs=40
    fi
    RUN_CONFIG=${RUN_CONFIG_DIR}/${MODEL_TYPE}/${TASK_NAME}_${CONFIG_NO}/${TASK_NAME}.json
    echo "output $OUTPUT_DIR Val interval $val_interval"

    # Generate config
    python jiant/jiant/proj/main/scripts/configurator.py \
        SimpleAPIMultiTaskConfigurator ${RUN_CONFIG} \
        --task_config_base_path ${DATA_DIR}/configs \
        --task_cache_base_path  ${CACHE_DIR}/${MODEL_TYPE} \
        --epochs $epochs \
        --train_batch_size $train_batch_size \
        --eval_batch_multiplier 2 \
        --train_task_name_list ${TRAIN_TASKS} \
        --val_task_name_list ${VAL_TASKS} \
        --gradient_accumulation_steps ${grad_acc} \
        --test_task_name_list  ${TASK_NAME} 
    
    if [[ ${TASK_NAME} == "adversarial_nli_r1" ]]; then
        python add_task_model_map.py ${RUN_CONFIG} $TASK_NAME
    fi

    if [[ ${TASK_NAME} == "mnli" ]]; then
        python add_task_model_map.py ${RUN_CONFIG} $TASK_NAME
    fi

    sbatch --export=DATA_DIR=$DATA_DIR,SEED=$SEED,MODELS_DIR=$MODELS_DIR,CACHE_DIR=$CACHE_DIR,RUN_CONFIG_DIR=${RUN_CONFIG},CONFIG_NO=${CONFIG_NO},OUTPUT_DIR=$OUTPUT_DIR,TASK_NAME=$TASK_NAME,MODEL_TYPE=$MODEL_TYPE,VAL_INTERVAL=$val_interval,LR=$lr jiant/irt_scripts/run_train_task.sbatch
}


function train_best_configs(){
    OUTPUT_DIR=${OUTPUT_DIR}_bestconfig
    run_training $MODEL_NAME boolq 2
    run_training $MODEL_NAME cb 8
    run_training $MODEL_NAME commonsenseqa 2
    run_training $MODEL_NAME copa 1
    run_training $MODEL_NAME rte 5
    run_training $MODEL_NAME wic 5
    run_training $MODEL_NAME snli 0
    run_training $MODEL_NAME qamr 0
    run_training $MODEL_NAME cosmosqa 1
    run_training $MODEL_NAME hellaswag 1
    run_training $MODEL_NAME wsc 4
    run_training $MODEL_NAME socialiqa 0
    run_training $MODEL_NAME arc_challenge 2
    run_training $MODEL_NAME arc_easy 2
    run_training $MODEL_NAME squad_v2 1
    run_training $MODEL_NAME arct 2
    run_training $MODEL_NAME mnli 0
    run_training $MODEL_NAME piqa 7
    run_training $MODEL_NAME mutual 1
    run_training $MODEL_NAME mutual_plus 1
    run_training $MODEL_NAME quoref 1
    run_training $MODEL_NAME mrqa_natural_questions 1
    run_training $MODEL_NAME newsqa 6
    run_training $MODEL_NAME mcscript 7
    run_training $MODEL_NAME mctaco 1
    run_training $MODEL_NAME quail 1
    run_training $MODEL_NAME winogrande 7
    run_training $MODEL_NAME abductive_nli 6
}

MODEL_NAME="${MODEL_TYPE##*/}"
OUTPUT_DIR=${OUTPUT_DIR}_${MODEL_NAME}

TASKMASTER_TASKS=(boolq cb commonsenseqa copa rte wic snli qamr cosmosqa hellaswag wsc socialiqa arc_challenge arc_easy squad_v2 arct mnli piqa mutual mutual_plus quoref mrqa_natural_questions newsqa mcscript mctaco quail winogrande abductive_nli)

