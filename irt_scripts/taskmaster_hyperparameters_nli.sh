WORKING_DIR=/scratch/cv50/jiant-2/experiments
DATA_DIR=${WORKING_DIR}/tasks
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache
RUN_CONFIG_DIR=${WORKING_DIR}/run_config_dir/taskmaster
OUTPUT_DIR=${WORKING_DIR}/output_dir/taskmaster

export PYTHONPATH=jiant/

MODEL_TYPE=$1
SEED=123456

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
  ["semi_auto_nli"]=1000
  ["nli_intervention"]=6000
)

BIG_TASKS=(adversarial_nli_r1 mnli mnli_mismatched squad_v2 record cosmosqa hellaswag abductive_nli arc_easy arc_challenge quoref newsqa mrqa_natural_questions mcscript mutual_plus mutual piqa quail)


function run_manual() {
    MODEL_TYPE=$1
    TASK_NAME=$2
    CONFIG_NO=$3
    SEED=1111


    if [[ " ${BIG_TASKS[*]} " == *" ${TASK_NAME} "* ]]; then
        train_batch_size=4
        grad_acc=4
    else
        train_batch_size=16
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
    TEST_TASKS=$TASK_NAME
    if [[ ${TASK_NAME} == "mnli" ]]; then
        VAL_TASKS="mnli,mnli_mismatched"
        TEST_TASKS="mnli,mnli_mismatched"
    elif [[ ${TASK_NAME} == "adversarial_nli_r1" ]]; then
        TRAIN_TASKS="adversarial_nli_r1,adversarial_nli_r2,adversarial_nli_r3,mnli,snli"
        VAL_TASKS="adversarial_nli_r1,adversarial_nli_r2,adversarial_nli_r3,mnli,mnli_mismatched,snli"
        TEST_TASKS="adversarial_nli_r1,adversarial_nli_r2,adversarial_nli_r3,mnli,mnli_mismatched,snli,nli_intervention,semi_auto_nli"
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
    echo $epochs

    python jiant/jiant/proj/main/scripts/configurator.py \
        SimpleAPIMultiTaskConfigurator ${RUN_CONFIG} \
        --task_config_base_path ${DATA_DIR}/configs \
        --task_cache_base_path  ${CACHE_DIR}/${MODEL_TYPE} \
        --epochs $epochs \
        --train_batch_size $train_batch_size \
        --eval_batch_multiplier 2 \
        --train_task_name_list ${TRAIN_TASKS} \
        --val_task_name_list ${VAL_TASKS} \
        --test_task_name_list ${TEST_TASKS} \
        --gradient_accumulation_steps ${grad_acc} 
    
    if [[ ${TASK_NAME} == "adversarial_nli_r1" ]]; then
        python add_task_model_map.py ${RUN_CONFIG} $TASK_NAME
    fi

    if [[ ${TASK_NAME} == "mnli" ]]; then
        python add_task_model_map.py ${RUN_CONFIG} $TASK_NAME
    fi

    sbatch --export=DATA_DIR=$DATA_DIR,SEED=$SEED,MODELS_DIR=$MODELS_DIR,CACHE_DIR=$CACHE_DIR,RUN_CONFIG_DIR=${RUN_CONFIG},CONFIG_NO=${CONFIG_NO},OUTPUT_DIR=$OUTPUT_DIR,TASK_NAME=$TASK_NAME,MODEL_TYPE=$MODEL_TYPE,VAL_INTERVAL=$val_interval,LR=$lr task.sbatch
}

function run_exp() {
    MODEL_TYPE=$1
    TASK_NAME=$2
    SEED=1111
    echo "$OUTPUT_DIR"

    echo "train size: ${TRAIN_SIZE[${TASK_NAME}]}"
    if ((  ${TRAIN_SIZE[${TASK_NAME}]} < 5000 )); then
        # use epoch 10 and 40
        CONFIG_NUMS=( 1 2 4 5 7 8 )
    else
        # use epoch 3 and 10
        CONFIG_NUMS=( 0 1 3 4 6 7 )
    fi


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

    if [[ ${TASK_NAME} == "abductive_nli" ]]; then
        grad_acc=1
    fi

    echo "${MODEL_TYPE} bsz ${train_batch_size} grad ${grad_acc}"


    ###
    VAL_TASKS=$TASK_NAME
    TRAIN_TASKS=$TASK_NAME
    TEST_TASKS=$TASK_NAME
    if [[ ${TASK_NAME} == "mnli" ]]; then
        VAL_TASKS="mnli,mnli_mismatched"
        TEST_TASKS="mnli,mnli_mismatched"
    elif [[ ${TASK_NAME} == "adversarial_nli_r1" ]]; then
        TRAIN_TASKS="adversarial_nli_r1,adversarial_nli_r2,adversarial_nli_r3,mnli,snli"
        VAL_TASKS="adversarial_nli_r1,adversarial_nli_r2,adversarial_nli_r3,mnli,mnli_mismatched,snli"
        TEST_TASKS="adversarial_nli_r1,adversarial_nli_r2,adversarial_nli_r3,mnli,mnli_mismatched,snli,nli_intervention,semi_auto_nli"
    fi
    ###

    for CONFIG_NO in ${CONFIG_NUMS[@]}
    do
        val_interval=$(expr ${TRAIN_SIZE[${TASK_NAME}]} / $train_batch_size)
        val_interval=$((val_interval>5000 ? 5000 : val_interval))
        echo "$TASK_NAME"
        echo "$TRAIN_TASKS"
        echo "$VAL_TASKS"
        echo "Val: ${val_interval}"
        echo "CONFIG: ${CONFIG_NO}"
        echo "Batch size: $train_batch_size"
        echo "Gradient Accumulation Steps: $grad_acc"


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
	      elif [[ ${CONFIG_NO} == 9 ]]; then
            lr=5e-5
            epochs=10
        fi

        RUN_CONFIG=${RUN_CONFIG_DIR}/${MODEL_TYPE}/${TASK_NAME}_${CONFIG_NO}/${TASK_NAME}.json 


        python jiant/jiant/proj/main/scripts/configurator.py \
           SimpleAPIMultiTaskConfigurator ${RUN_CONFIG} \
           --task_config_base_path ${DATA_DIR}/configs \
           --task_cache_base_path  ${CACHE_DIR}/${MODEL_TYPE} \
           --epochs $epochs \
           --train_batch_size $train_batch_size \
           --eval_batch_multiplier 2 \
           --train_task_name_list ${TRAIN_TASKS} \
           --val_task_name_list ${VAL_TASKS} \
           --test_task_name_list ${TEST_TASKS} \
	         --gradient_accumulation_steps ${grad_acc}

        echo "Done"	
      	if [[ ${TASK_NAME} == "adversarial_nli_r1" ]]; then
                  python add_task_model_map.py ${RUN_CONFIG} $TASK_NAME
              fi

              if [[ ${TASK_NAME} == "mnli" ]]; then
                  python add_task_model_map.py ${RUN_CONFIG} $TASK_NAME
              fi

      sbatch --export=DATA_DIR=$DATA_DIR,SEED=$SEED,MODELS_DIR=$MODELS_DIR,CACHE_DIR=$CACHE_DIR,RUN_CONFIG_DIR=${RUN_CONFIG},CONFIG_NO=${CONFIG_NO},OUTPUT_DIR=$OUTPUT_DIR,TASK_NAME=$TASK_NAME,MODEL_TYPE=$MODEL_TYPE,VAL_INTERVAL=$val_interval,LR=$lr task.sbatch
    done
}

function download_data() {
    TASKMASTER_TASKS=$1
    for TASK_NAME in "${TASKMASTER_TASKS[@]}"
    do
       echo "Downloading $TASK_NAME"
       python jiant/jiant/scripts/download_data/runscript.py download --tasks $TASK_NAME --output_path /scratch/cv50/jiant-2/experiments/tasks/ 
   done
}

function prepare_data() {
   TASKMASTER_TASKS=$2
   for TASK_NAME in "${TASKMASTER_TASKS[@]}"
   do
       echo "run preprocess $TASK_NAME"
       bash run_preprocess.sh $1 $TASK_NAME
   done
}

function tune_hyperparameters() {
   TASKMASTER_TASKS=$2
   for TASK_NAME in "${TASKMASTER_TASKS[@]}"
   do  
       echo "run exp"
       run_exp $1 $TASK_NAME
   done 
}


MODEL_NAME="${MODEL_TYPE##*/}"
OUTPUT_DIR=${OUTPUT_DIR}_${MODEL_NAME}

#TASKMASTER_TASKS=(boolq cb commonsenseqa copa cosmosqa hellaswag mnli rte socialiqa wic snli wnli qamr adversarial_nli_r1 squad_v2 abductive_nli)
# TASKMASTER_TASKS=(squad_v2)

# TASKMASTER_TASKS=(mnli mnli_mismatched snli)

# download_data $TASKMASTER_TASKS
# prepare_data $MODEL_TYPE $TASKMASTER_TASKS
# tune_hyperparameters $MODEL_NAME $TASKMASTER_TASKS


# OUTPUT_DIR=${OUTPUT_DIR}
# OUTPUT_DIR=${OUTPUT_DIR}_bestconfig_${SEED}
# run_manual $MODEL_NAME boolq 2
# run_manual $MODEL_NAME cb 8
# run_manual $MODEL_NAME commonsenseqa 2
# run_manual $MODEL_NAME copa 1
# run_manual $MODEL_NAME rte 5
# run_manual $MODEL_NAME rte 8
# run_manual $MODEL_NAME wic 5
# run_manual $MODEL_NAME snli 0
# run_manual $MODEL_NAME qamr 0
# run_manual $MODEL_NAME cosmosqa 1
# run_manual $MODEL_NAME hellaswag 1
# run_manual $MODEL_NAME wsc 4
# run_manual $MODEL_NAME socialiqa 0
# run_manual $MODEL_NAME arc_challenge 2
# run_manual $MODEL_NAME arc_easy 2
# run_manual $MODEL_NAME squad_v2 1
# run_manual $MODEL_NAME arct 2
# run_manual $MODEL_NAME mnli_mismatched 0
# run_manual $MODEL_NAME mnli 0
# run_manual $MODEL_NAME piqa 7
# run_manual $MODEL_NAME mutual 1
# run_manual $MODEL_NAME mutual_plus 1
# run_manual $MODEL_NAME quoref 1
# run_manual $MODEL_NAME mrqa_natural_questions 1
# run_manual $MODEL_NAME newsqa 6
# run_manual $MODEL_NAME mcscript 7
# run_manual $MODEL_NAME mctaco 1
# run_manual $MODEL_NAME mctest160 2
# run_manual $MODEL_NAME mctest500 8
# run_manual $MODEL_NAME quail 1
# run_manual $MODEL_NAME winogrande 7
# run_manual $MODEL_NAME abductive_nli 6
run_manual $MODEL_NAME adversarial_nli_r1 0
