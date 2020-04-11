# Usage:
# First, pretrain on the intermediate tasks and save the file.
# run first_intermediate_exp <intermediate_task_name>
# for example, <intermediate_task_name> mnli
# Then, for each of the target tasks and/or probing tasks, run
# run_intermediate_to_target_task <intemeidate_task> <target_task> <directory_to_jiant>
# for example, run_intermediate_to_target_task  mnli copa /beegfs/yp913/jiant/transfer_analysis
# IMPORTANT!! Make sure the configs for your task is correct in config/taskmaster/base_roberta.conf
# to look at ways to easily specify task-specific configs for your task, look at
# glue-small-tasks-tmpl-2 in defaults.conf

function run_exp() {
    # Helper function to invoke main.py.
    # Don't run this directly - use the experiment functions below,
    # or create a new one for a new experiment suite.
    # Usage: run_exp <config_file> <overrides> <hparam_set> <random_seed>
    CONFIG_FILE=$1
    OVERRIDES=$2
    RANDOM_SEED=${3:-1234}
    # Add random seed
    OVERRIDES+=", random_seed=${RANDOM_SEED}"
    OVERRIDES+=", cuda=\"auto\""
    echo ${OVERRIDES}
    sbatch sb_hellaswag.sh "${CONFIG_FILE}" "${OVERRIDES}" 
}


declare -A INTERM_LR=(
  ["sst"]=0.00001
  ["SocialIQA"]=0.00001
  ["qqp"]=0.00001
  ["mnli"]=0.00002
  ["scitail"]=0.00001
  ["qasrl"]=0.00002
  ["qamr"]=0.00002
  ["squad"]=5
  ["cosmosqa"]=0.00001
  ["ccg"]=0.00001
  ["commonsenseqa"]=0.000003
  # === Target
)
declare -A INTERM_BSIZE=(
  ["sst"]=8
  ["qqp"]=16
  ["commonsenseqa"]=8
)

declare -A INTERM_MEPOCH=(
  ["sst"]=7
  ["qqp"]=15
  ["commonsenseqa"]=15
  )

declare -A INTERM_DATA_FRACTION=(
    ["qqp"]="0.0549"
    ["commonsenseqa"]="1.0" # this is the smallest one. 
    ["sst"]="0.29696"
)

###
# Why we chose these target tasks for macro_avg
#For commonsesneqa - wic (didn’t improve that much), Copa (improve da lot), WSC (decreased slightly)
#For QQP - commonsenseaqQA did BAD, RTE was slightly negative  stayed natural, and win increased. 
#For SST -  copa was neutral., multirc sucked, RTE increased A LOT. 
###
declare -A TARGET_TASK_FOR_INTERM=(
  ["commonsenseqa"]={"wic" "copa" "winograd-coreference"}
  ["qqp"]={"commonsenseqa" "rte-superglue" "wic"}
  ["sst"]={"copa" "multirc" "rte-superglue"}
)

#########################################
# Hyperparameter tuning experiments
#########################################

function hyperparameter_mlm_K_sweep_interm_train() {
    # Do hyerparameter tuning search for the parameters
    # Usage: hyperparameter_sweep <task> <random_seed>
    for K in 1048576 131072 16384 # 2^20, 2^17, and 2^14. 
    do
        OVERRIDES=" run_name=\"$1_mlm_new_$2_$K\", exp_name=roberta-large-K, pretrain_tasks=\"$1,mlm\", target_tasks=\"\",reload_tasks=0, reload_indexing=0, reload_vocab=0, weighting_method=\"examples-proportional-mixingK=$K\", input_module=roberta-large, do_target_task_training=0, transfer_paradigm=finetune, early_stopping_method=$1, do_pretrain=1"
        EXP_OVERRIDES="${OVERRIDES}, run_name=\"$1_mlm_new_$2_$K\", pretrain_data_fraction=${INTERM_DATA_FRACTION[$1]}, lr=${INTERM_LR[$1]}, batch_size=${INTERM_BSIZE[$1]}, max_epochs=${INTERM_MEPOCH[$1]}"
        TASK_TYPE="regular"
        if [[ ${TASK_TYPE} == "edge" ]]; then
            BASE_CONFIG_FILE="base_edgeprobe"
        elif [[ ${TASK_TYPE} == "regular" ]]; then
            BASE_CONFIG_FILE="base_albert"
        fi
        run_exp "jiant/config/taskmaster/clean_roberta.conf" "${EXP_OVERRIDES}" $2
    done
}


function hyperparameter_mlm_K_sweep_interm_only_train() {
    # Do hyerparameter tuning search for the parameters
    # Usage: hyperparameter_sweep <task> <random_seed>
    for K in 1048576 131072 16384 # 2^20, 2^17, and 2^14. 
    do
        OVERRIDES=" run_name=\"$1_new_$2_noK\", exp_name=roberta-large-K-no-mlm, pretrain_tasks=\"$1\", target_tasks=\"\",reload_tasks=0, reload_indexing=0, reload_vocab=0, input_module=roberta-large, do_target_task_training=0, transfer_paradigm=finetune, early_stopping_method=$1, do_pretrain=1"
        EXP_OVERRIDES="${OVERRIDES}, run_name=\"$1_new_$2_noK\", pretrain_data_fraction=${INTERM_DATA_FRACTION[$1]}, lr=${INTERM_LR[$1]}, batch_size=${INTERM_BSIZE[$1]}, max_epochs=${INTERM_MEPOCH[$1]}"
        TASK_TYPE="regular"
        if [[ ${TASK_TYPE} == "edge" ]]; then
            BASE_CONFIG_FILE="base_edgeprobe"
        elif [[ ${TASK_TYPE} == "regular" ]]; then
            BASE_CONFIG_FILE="base_albert"
        fi
        run_exp "jiant/config/taskmaster/clean_roberta.conf" "${EXP_OVERRIDES}" $2
    done
}



function run_p1_specific() {
  # Do hyerparameter tuning search for the parameters
  # Usage: hyperparameter_sweep <task> <random_seed>
   OVERRIDES=" run_name=\"$1_mlm_new_$2_$3\", exp_name=roberta-large-K, pretrain_tasks=\"$1,mlm\", target_tasks=\"\",reload_tasks=0, reload_indexing=0, reload_vocab=0, weighting_method=\"examples-proportional-mixingK=$3\", input_module=roberta-large, do_target_task_training=0, transfer_paradigm=finetune, early_stopping_method=$1, do_pretrain=1"
   EXP_OVERRIDES="${OVERRIDES}, run_name=\"$1_mlm_new_$2_$3\", pretrain_data_fraction=${INTERM_DATA_FRACTION[$1]}, lr=${INTERM_LR[$1]}, batch_size=8, max_epochs=${INTERM_MEPOCH[$1]}"
   TASK_TYPE="regular"
   if [[ ${TASK_TYPE} == "edge" ]]; then
     BASE_CONFIG_FILE="base_edgeprobe"
   elif [[ ${TASK_TYPE} == "regular" ]]; then
     BASE_CONFIG_FILE="base_albert"
   fi
   run_exp "jiant/config/taskmaster/clean_roberta.conf" "${EXP_OVERRIDES}" $2
}

hyperparameter_mlm_K_sweep_interm_only_train commonsenseqa 5238211
#run_p1_specific qqp 5238211 16384
#run_p1_specific qqp 5238211 131072
#hyperparameter_mlm_K_sweep_interm_train commonsenseqa 111001  
#hyperparameter_mlm_K_sweep_interm_train commonsenseqa 5238211
#hyperparameter_mlm_K_sweep_interm_train commonsenseqa  921
