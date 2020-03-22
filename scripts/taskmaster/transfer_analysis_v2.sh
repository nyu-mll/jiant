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
    RANDOM_SEED=${4:-1234}

    if [[ $3 == 0 ]]; then
        OVERRIDES+=", lr=2e-5, dropout=0.2"
    elif [[ $3 == 1 ]]; then
        OVERRIDES+=", lr=2e-5, dropout=0.1"
    elif [[ $3 == 2 ]]; then
        OVERRIDES+=", lr=1e-5, dropout=0.2"
    elif [[ $3 == 3 ]]; then
        OVERRIDES+=", lr=1e-5, dropout=0.1"
    elif [[ $3 == 4 ]]; then
        OVERRIDES+=", lr=5e-6, dropout=0.2"
    elif [[ $3 == 5 ]]; then
        OVERRIDES+=", lr=5e-6, dropout=0.1"
    elif [[ $3 == 6 ]]; then
        OVERRIDES+=", lr=3e-6, dropout=0.2"
    elif [[ $3 == 7 ]]; then
        OVERRIDES+=", lr=3e-6, dropout=0.1"
    elif [[ $3 == 8 ]]; then
        OVERRIDES+=", lr=1e-3, dropout=0.2"
    elif [[ $3 == 9 ]]; then
        OVERRIDES+=", lr=1e-3, dropout=0.1"
    elif [[ $3 == 10 ]]; then
        OVERRIDES+=", lr=1e-4, dropout=0.2"
    elif [[ $3 == 11 ]]; then
        OVERRIDES+=", lr=1e-4, dropout=0.1"
    fi

    # Add random seed
    OVERRIDES+=", random_seed=${RANDOM_SEED}"
    OVERRIDES+=", batch_size=2, accumulation_steps=2"

    # Construct args
    declare -a args
    args+=( --config_file "${CONFIG_FILE}" )
    args+=( -o "${OVERRIDES}" )
   
    echo "${CONFIG_FILE}"
    echo "${OVERRIDES}"
    # Run
    #python main.py "${args[@]}"
    echo "${OVERRIDES}"
    #sbatch sb_hellaswag.sh "${CONFIG_FILE}" "${OVERRIDES}"
}

declare -A TASK_TYPE_MAP
TASK_TYPE_MAP["edges-ner-ontonotes"]="edge"
TASK_TYPE_MAP["edges-srl-ontonotes"]="edge"
TASK_TYPE_MAP["edges-coref-ontonotes"]="edge"
TASK_TYPE_MAP["edges-spr1"]="edge"
TASK_TYPE_MAP["edges-spr2"]="edge"
TASK_TYPE_MAP["edges-dpr"]="edge"
TASK_TYPE_MAP["edges-rel-semeval"]="edge"
TASK_TYPE_MAP["edges-pos-ontonotes"]="edge"
TASK_TYPE_MAP["edges-nonterminal-ontonotes"]="edge"
TASK_TYPE_MAP["edges-dep-ud-ewt"]="edge"
TASK_TYPE_MAP["se-probing-word-content"]="regular"
TASK_TYPE_MAP["se-probing-tree-depth"]="regular"
TASK_TYPE_MAP["se-probing-top-constituents"]="regular"
TASK_TYPE_MAP["se-probing-bigram-shift"]="regular"
TASK_TYPE_MAP["se-probing-past-present"]="regular"
TASK_TYPE_MAP["se-probing-subj-number"]="regular"
TASK_TYPE_MAP["se-probing-obj-number"]="regular"
TASK_TYPE_MAP["se-probing-odd-man-out"]="regular"
TASK_TYPE_MAP["se-probing-coordination-inversion"]="regular"
TASK_TYPE_MAP["se-probing-sentence-length"]="regular"
TASK_TYPE_MAP["acceptability-wh"]="regular"
TASK_TYPE_MAP["acceptability-def"]="regular"
TASK_TYPE_MAP["acceptability-conj"]="regular"
TASK_TYPE_MAP["acceptability-eos"]="regular"
TASK_TYPE_MAP["cola"]="regular"

declare -A INTERM_HPARAM=(
  ["sst"]=7
  ["SocialIQA"]=0
  ["qqp"]=5
  ["mnli"]=6
  ["scitail"]=4
  ["qasrl"]=7
  ["qamr"]=3
  ["squad"]=5
  ["cosmosqa"]=6
  ["hellaswag"]=6
  ["commonsenseqa"]=6
  ["ccg"]=5
  # === Target
  ["rte-superglue"]=4
  ["boolq"]=7
  ["commitbank"]=2
  ["copa"]=5
  ["multirc"]=0
  ["record"]=2
  ["wic"]=3
  ["winograd-coreference"]=5
)
declare -A INTERM_BSIZE=(
  ["sst"]=64
  ["SocialIQA"]=4
  ["qqp"]=8
  ["mnli"]=4
  ["scitail"]=4
  ["qasrl"]=4
  ["qamr"]=4
  ["squad"]=4
  ["cosmosqa"]=4
  ["hellaswag"]=4
  ["commonsenseqa"]=4
  ["ccg"]=4
  ["mlm"]=2
  # === Target
  ["rte-superglue"]=4
  ["boolq"]=4
  ["commitbank"]=4
  ["copa"]=32
  ["multirc"]=4
  ["record"]=4
  ["wic"]=32
  ["winograd-coreference"]=32
)
declare -A TARGET_HPARAM=(
  ["rte-superglue"]=4
  ["boolq"]=7
  ["commitbank"]=2
  ["copa"]=5
  ["multirc"]=0
  ["record"]=2
  ["wic"]=3
  ["winograd-coreference"]=5
  ["commonsenseqa"]=6
  ["cosmosqa"]=7
)
declare -A TARGET_BSIZE=(
  ["rte-superglue"]=4
  ["boolq"]=4
  ["commitbank"]=4
  ["copa"]=32
  ["multirc"]=4
  ["record"]=4
  ["wic"]=32
  ["winograd-coreference"]=32
  ["commonsenseqa"]=4
  ["cosmosqa"]=4
)
declare -A PROBING_HPARAM=(
  ["edges-ner-ontonotes"]=0
  ["edges-srl-ontonotes"]=0
  ["edges-coref-ontonotes"]=2
  ["edges-spr1"]=1
  ["edges-spr2"]=5
  ["edges-dpr"]=5
  ["edges-rel-semeval"]=1
  ["se-probing-word-content"]=0
  ["se-probing-tree-depth"]=1
  ["se-probing-top-constituents"]=0
  ["se-probing-bigram-shift"]=2
  ["se-probing-past-present"]=2
  ["se-probing-subj-number"]=0
  ["se-probing-obj-number"]=4
  ["se-probing-odd-man-out"]=3
  ["se-probing-coordination-inversion"]=0
  ["edges-pos-ontonotes"]=1
  ["edges-nonterminal-ontonotes"]=4
  ["edges-dep-ud-ewt"]=0
  ["se-probing-sentence-length"]=0
  ["acceptability-wh"]=3
  ["acceptability-def"]=0
  ["acceptability-conj"]=1
  ["acceptability-eos"]=3
  ["cola"]=5
)
declare -A PROBING_BSIZE=(
  ["edges-ner-ontonotes"]=8
  ["edges-srl-ontonotes"]=8
  ["edges-coref-ontonotes"]=8
  ["edges-spr1"]=32
  ["edges-spr2"]=4
  ["edges-dpr"]=4
  ["edges-rel-semeval"]=4
  ["se-probing-word-content"]=64
  ["se-probing-tree-depth"]=64
  ["se-probing-top-constituents"]=64
  ["se-probing-bigram-shift"]=64
  ["se-probing-past-present"]=64
  ["se-probing-subj-number"]=64
  ["se-probing-obj-number"]=64
  ["se-probing-odd-man-out"]=64
  ["se-probing-coordination-inversion"]=64
  ["edges-pos-ontonotes"]=8
  ["edges-nonterminal-ontonotes"]=8
  ["edges-dep-ud-ewt"]=4
  ["se-probing-sentence-length"]=64
  ["acceptability-wh"]=4
  ["acceptability-def"]=4
  ["acceptability-conj"]=4
  ["acceptability-eos"]=4
  ["cola"]=4
)
declare -A MIXING_HPARAM=(
  ["edges-ner-ontonotes"]=1
  ["edges-srl-ontonotes"]=0
  ["edges-coref-ontonotes"]=1
  ["edges-spr1"]=1
  ["edges-spr2"]=1
  ["edges-dpr"]=0
  ["edges-rel-semeval"]=1
  ["se-probing-word-content"]=9
  ["se-probing-tree-depth"]=9
  ["se-probing-top-constituents"]=8
  ["se-probing-bigram-shift"]=9
  ["se-probing-past-present"]=2
  ["se-probing-subj-number"]=9
  ["se-probing-obj-number"]=9
  ["se-probing-odd-man-out"]=9
  ["se-probing-coordination-inversion"]=1
  ["edges-pos-ontonotes"]=8
  ["edges-nonterminal-ontonotes"]=9
  ["edges-dep-ud-ewt"]=8
  ["se-probing-sentence-length"]=9
  ["acceptability-wh"]=8
  ["acceptability-def"]=8
  ["acceptability-conj"]=8
  ["acceptability-eos"]=7
  ["cola"]=9
)
declare -A MIXING_BSIZE=(
  ["edges-ner-ontonotes"]=16
  ["edges-srl-ontonotes"]=16
  ["edges-coref-ontonotes"]=16
  ["edges-spr1"]=16
  ["edges-spr2"]=16
  ["edges-dpr"]=16
  ["edges-rel-semeval"]=16
  ["se-probing-word-content"]=128
  ["se-probing-tree-depth"]=128
  ["se-probing-top-constituents"]=128
  ["se-probing-bigram-shift"]=128
  ["se-probing-past-present"]=128
  ["se-probing-subj-number"]=128
  ["se-probing-obj-number"]=128
  ["se-probing-odd-man-out"]=128
  ["se-probing-coordination-inversion"]=128
  ["edges-pos-ontonotes"]=128
  ["edges-nonterminal-ontonotes"]=128
  ["edges-dep-ud-ewt"]=128
  ["se-probing-sentence-length"]=16
  ["acceptability-wh"]=4
  ["acceptability-def"]=4
  ["acceptability-conj"]=4
  ["acceptability-eos"]=4
  ["cola"]=16
)
declare -A SEED_DICT=(
  ["run1_intermediate"]=1111001
  ["run1_stilts"]=1111002
  ["run1_mixing"]=1111003
  ["run1_probing"]=1111003
  ["run2_intermediate"]=523821
  ["run2_stilts"]=523822
  ["run2_mixing"]=523823
  ["run2_probing"]=523823
  ["run3_intermediate"]=921
  ["run3_stilts"]=922
  ["run3_mixing"]=923
  ["run3_probing"]=923
)
declare -A INTERM_DATA_FRACTION=(
    ["ccgS1"]="0.032882"
    ["ccgS2"]="0.065764"
    ["ccgS3"]="0.131527"
    ["ccgS4"]="0.263054"
    ["ccgS5"]="0.526108"
    ["ccgS6"]="1.0"
    ["qqpS1"]="0.003436"
    ["qqpS2"]="0.006871"
    ["qqpS3"]="0.013742"
    ["qqpS4"]="0.027484"
    ["qqpS5"]="0.054968"
    ["qqpS6"]="0.109937"
    ["qqpS7"]="0.219873"
    ["qqpS8"]="0.439746"
    ["qqpS9"]="0.879493"
    ["qqpS10"]="1.0"
    ["hellaswagS1"]="0.031324"
    ["hellaswagS2"]="0.062649"
    ["hellaswagS3"]="0.125298"
    ["hellaswagS4"]="0.250595"
    ["hellaswagS5"]="0.501190"
    ["hellaswagS6"]="1.0"
    ["mnliS1"]="0.003183"
    ["mnliS2"]="0.006366"
    ["mnliS3"]="0.012732"
    ["mnliS4"]="0.025465"
    ["mnliS5"]="0.050929"
    ["mnliS6"]="0.101858"
    ["mnliS7"]="0.203717"
    ["mnliS8"]="0.407434"
    ["mnliS9"]="0.814867"
    ["mnliS10"]="1.0"
    ["commonsenseqaS1"]="0.128324"
    ["commonsenseqaS2"]="0.256647"
    ["commonsenseqaS3"]="0.513294"
    ["commonsenseqaS4"]="1.0"
)
export TM_TARGET_TASK_NAMES=(rte-superglue boolq commitbank copa multirc record wic winograd-coreference commonsenseqa cosmosqa)
export TM_PROBING_TASK_NAMES=(edges-ner-ontonotes edges-srl-ontonotes edges-coref-ontonotes edges-spr1 edges-spr2 edges-dpr edges-rel-semeval se-probing-word-content se-probing-tree-depth se-probing-top-constituents se-probing-bigram-shift se-probing-past-present se-probing-subj-number se-probing-obj-number se-probing-odd-man-out se-probing-coordination-inversion edges-pos-ontonotes edges-nonterminal-ontonotes edges-dep-ud-ewt se-probing-sentence-length acceptability-wh acceptability-def acceptability-conj acceptability-eos cola)
export TM_MIXING_TASK_NAMES=(edges-ner-ontonotes edges-srl-ontonotes edges-coref-ontonotes edges-spr1 edges-spr2 edges-dpr edges-rel-semeval se-probing-word-content se-probing-tree-depth se-probing-top-constituents se-probing-bigram-shift se-probing-past-present se-probing-subj-number se-probing-obj-number se-probing-odd-man-out se-probing-coordination-inversion edges-pos-ontonotes edges-nonterminal-ontonotes edges-dep-ud-ewt se-probing-sentence-length acceptability-wh acceptability-def acceptability-conj acceptability-eos cola)



#########################################
# Hyperparameter tuning experiments
#########################################
function hyperparameter_sweep() {
    # Do hyerparameter tuning search for the parameters
    # Usage: hyperparameter_sweep <task> <batch_size> <random_seed>
    OVERRIDES="exp_name=roberta-large"
    OVERRIDES+=", target_tasks=$1, do_pretrain=0, batch_size=$2, reload_vocab=1, do_target_task_training=1, input_module=roberta-large,pretrain_tasks=\"\""
    for i in 0 1 2 3 4 5 6 7
    do
        EXP_OVERRIDES="${OVERRIDES}, run_name=$1config$i"
        TASK_TYPE=${TASK_TYPE_MAP[$1]}
        if [[ ${TASK_TYPE} == "edge" ]]; then
            BASE_CONFIG_FILE="base_edgeprobe"
        elif [[ ${TASK_TYPE} == "regular" ]]; then
            BASE_CONFIG_FILE="base_roberta"
        fi
        run_exp "jiant/config/taskmaster/${BASE_CONFIG_FILE}.conf" "${EXP_OVERRIDES}" $i $3
    done

}

function hyperparameter_sweep_mix() {
    # Do hyerparameter tuning search for the parameters
    # Usage: hyperparameter_sweep <task> <batch_size> <random_seed>
    OVERRIDES="exp_name=roberta-large"
    OVERRIDES+=", target_tasks=$1, do_pretrain=0, batch_size=$2, reload_vocab=1, transfer_paradigm=frozen, allow_untrained_encoder_parameters=1, pytorch_transformers_output_mode = mix,do_target_task_training=1, input_module=roberta-large,pretrain_tasks=\"\""
    for i in 0 1 2 3 4 5 6 7 8 9
    do
        EXP_OVERRIDES="${OVERRIDES}, run_name=$1configmix$i"
        TASK_TYPE=${TASK_TYPE_MAP[$1]}
        if [[ ${TASK_TYPE} == "edge" ]]; then
            BASE_CONFIG_FILE="base_edgeprobe"
        elif [[ ${TASK_TYPE} == "regular" ]]; then
            BASE_CONFIG_FILE="base_roberta"
        fi
        run_exp "jiant/config/taskmaster/${BASE_CONFIG_FILE}.conf" "${EXP_OVERRIDES}" $i $3
    done

}

#########################################
# Functions relating to main experiments
#########################################

function first_intermediate_exp() {
    # Initial intermediate task pretraining.
    # Usage: first_intermediate_task <intermediate_task_name> <config_number> <batch_size> <random_seed> <run_number>
    OVERRIDES="exp_name=roberta-large, run_name=$1_$5_$2, batch_size=$3, reload_vocab=1"
    OVERRIDES+=", target_tasks=$1, do_pretrain=1, do_target_task_training=0, input_module=roberta-large,pretrain_tasks=$1"
    OVERRIDES+=", do_full_eval=1"
    run_exp "jiant/config/taskmaster/base_roberta.conf" "${OVERRIDES}" ${2} ${4}
}

function first_intermediate_exp_limited_size() {
    # Initial intermediate task pretraining with limited size.
    # Usage: first_intermediate_task <intermediate_task_name> <config_number> <batch_size> <random_seed> <run_number>
    # <intermediate_task_name> should use _size to seperate real task name and size, e.g. ccgS1
    IFS="S" read -ra ADDR <<< "${1}"
    TASK_NAME=${ADDR[0]}
    OVERRIDES="exp_name=roberta-large, run_name=$1_$5, batch_size=$3, reload_vocab=1"
    OVERRIDES+=", target_tasks=$TASK_NAME, do_pretrain=1, do_target_task_training=0, input_module=roberta-large,pretrain_tasks=$TASK_NAME"
    OVERRIDES+=", do_full_eval=1, pretrain_data_fraction=${INTERM_DATA_FRACTION[$1]}"
    run_exp "jiant/config/taskmaster/base_roberta.conf" "${OVERRIDES}" ${2} ${4}
}

function first_target_exp() {
    # Initial intermediate task pretraining.
    # Usage: first_intermediate_task <intermediate_task_name> <config_number> <batch_size> <random_seed> <run_number>
    OVERRIDES="exp_name=roberta-large, run_name=$1_$5, batch_size=$3, reload_vocab=1"
    OVERRIDES+=", target_tasks=$1, do_pretrain=0, do_target_task_training=1, input_module=roberta-large,pretrain_tasks=$1"
    OVERRIDES+=", do_full_eval=1"
    run_exp "jiant/config/taskmaster/base_roberta.conf" "${OVERRIDES}" ${2} ${4}
}
function run_intermediate_to_target_task() {
    # Using a pretrained intermediate task, finetune on a target task.  ("STILTs" sheet)
    # This function can also be used to finetune on a probing task as well.
    # Usage: run_intermediate_to_target_task <intermediate_task> <target_task> <directory_to_project_dir> <config_number> <batch_size> <random_seed> <run>
    OVERRIDES="exp_name=$1, run_name=$2_run$7"
    OVERRIDES+=", target_tasks=$2, load_model=1, load_target_train_checkpoint=$3/roberta-large/$1_$7/model_*.best.th, pretrain_tasks=\"\""
    OVERRIDES+=", input_module=roberta-large, batch_size=$5, reload_vocab=1"
    OVERRIDES+=", do_pretrain=0, do_target_task_training=1"
    run_exp "jiant/config/taskmaster/base_roberta.conf" "${OVERRIDES}" ${4} ${6}
}

function run_intermediate_to_probing() {
    # Using a pretrained intermediate task, finetune on an probing task.  ("Probing" sheet)
    # Usage: run_intermediate_to_probing <intermediate_task> <probing task> <directory_to_project_dir> <config_number> <batch_size> <random_seed> <run>
    OVERRIDES="exp_name=$1, run_name=$2_run$7"
    OVERRIDES+=", target_tasks=$2, load_model=1, load_target_train_checkpoint=$3/roberta-large/$1_$7/model_*.best.th, pretrain_tasks=\"\""
    OVERRIDES+=", input_module=roberta-large, batch_size=$5, reload_vocab=1"
    OVERRIDES+=", do_pretrain=0, do_target_task_training=1"
    TASK_TYPE=${TASK_TYPE_MAP[$2]}
    if [[ ${TASK_TYPE} == "edge" ]]; then
        BASE_CONFIG_FILE="base_edgeprobe"
    elif [[ ${TASK_TYPE} == "regular" ]]; then
        BASE_CONFIG_FILE="base_roberta"
    fi
    run_exp "jiant/config/taskmaster/${BASE_CONFIG_FILE}.conf" "${OVERRIDES}" ${4} ${6}
}

function run_intermediate_to_mixing() {
    # Using a pretrained intermediate task, use frozen encoder with mixing on an probing task.  ("Mixing" sheet)
    # Usage: run_intermediate_to_mixing <intermediate_task> <probing task> <directory_to_project_dir> <config_number> <batch_size> <random_seed> <run>
    OVERRIDES="exp_name=$1, run_name=$2_mixrun$7"
    OVERRIDES+=", target_tasks=$2, load_model=1, load_target_train_checkpoint=$3/roberta-large/$1_$7/model_*.best.th, pretrain_tasks=\"\""
    OVERRIDES+=", input_module=roberta-large, batch_size=$5, reload_vocab=1"
    OVERRIDES+=", transfer_paradigm=frozen, allow_untrained_encoder_parameters=1, pytorch_transformers_output_mode = mix"
    OVERRIDES+=", do_pretrain=0, do_target_task_training=1"
    run_exp "jiant/config/taskmaster/base_edgeprobe.conf" "${OVERRIDES}" ${4} ${6}
}
# ez_run_intermediate_to_probing 2 mnli edges-dpr /beegfs/yp913/jiant/coreference_exp

#########################################
# EASY Functions
#########################################

function ez_first_intermediate_exp() {
    # Usage: ez_first_intermediate_exp <1:run_num> <2:intermediate_task>
    first_intermediate_exp ${2} ${INTERM_HPARAM[$2]}  ${INTERM_BSIZE[${2}]} ${SEED_DICT[run${1}_intermediate]} ${1}
}

function ez_first_intermediate_exp_limited_size() {
    # Usage: ez_first_intermediate_exp_limited_size <1:run_num> <2:intermediate_task>
    # <intermediate_task_name> should use _size to seperate real task name and size, e.g. ccgS1
    IFS="S" read -ra ADDR <<< "${2}"
    TASK_NAME=${ADDR[0]}
    first_intermediate_exp_limited_size ${2} ${INTERM_HPARAM[$TASK_NAME]} ${INTERM_BSIZE[$TASK_NAME]} ${SEED_DICT[run${1}_intermediate]} ${1}
}

function ez_first_target_exp() {
    # Usage: ez_first_intermediate_exp <1:run_num> <2:intermediate_task>
    first_target_exp ${2} ${INTERM_HPARAM[${2}]} ${INTERM_BSIZE[${2}]} ${SEED_DICT[run${1}_intermediate]} ${1}
}

function ez_run_intermediate_to_target_task() {
    # Usage: ez_run_intermediate_to_target_task <1:run_num> <2:intermediate_task> <3:target_task> <4:directory_to_project_dir>
    run_intermediate_to_target_task ${2} ${3} ${4} ${TARGET_HPARAM[${3}]} ${TARGET_BSIZE[${3}]} ${SEED_DICT[run${1}_stilts]} ${1}
}

function ez_run_intermediate_to_probing() {
    # Usage: ez_run_intermediate_to_probing <1:run_num> <2:intermediate_task> <3:probing_task> <4:directory_to_project_dir>
    run_intermediate_to_probing ${2} ${3} ${4} ${PROBING_HPARAM[${3}]} ${PROBING_BSIZE[${3}]} ${SEED_DICT[run${1}_probing]} ${1}
}

function ez_run_intermediate_to_mixing() {
    # Usage: ez_run_intermediate_to_mixing <1:run_num> <2:intermediate_task> <3:mixing_task> <4:directory_to_project_dir>
    run_intermediate_to_mixing ${2} ${3} ${4} ${MIXING_HPARAM[${3}]} ${MIXING_BSIZE[${3}]} ${SEED_DICT[run${1}_mixing]} ${1}
}
