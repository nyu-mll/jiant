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

    if [ $3 == 0 ]; then
        OVERRIDES+=", lr=2e-5, dropout=0.2"
    elif [ $3 == 1 ]; then
        OVERRIDES+=", lr=2e-5, dropout=0.1"
    elif [ $3 == 2 ]; then
        OVERRIDES+=", lr=1e-5, dropout=0.2"
    elif [ $3 == 3 ]; then
        OVERRIDES+=", lr=1e-5, dropout=0.1"
    elif [ $3 == 4 ]; then
        OVERRIDES+=", lr=5e-6, dropout=0.2"
    elif [ $3 == 5 ]; then
        OVERRIDES+=", lr=5e-6, dropout=0.1"
    elif [ $3 == 6 ]; then
        OVERRIDES+=", lr=3e-6, dropout=0.2"
    elif [ $3 == 7 ]; then
        OVERRIDES+=", lr=3e-6, dropout=0.1"
    elif [ $3 == 8 ]; then
        OVERRIDES+=", lr=1e-3, dropout=0.2"
    elif [ $3 == 9 ]; then
        OVERRIDES+=", lr=1e-3, dropout=0.1"
    fi

    # Add random seed
    OVERRIDES+=", random_seed=${RANDOM_SEED}}"

    # Construct args
    declare -a args
    args+=( --config_file "${CONFIG_FILE}" )
    args+=( -o "${OVERRIDES}" )

    # Run
    python main.py "${args[@]}"
}

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
        run_exp "jiant/config/taskmaster/base_roberta.conf" "${EXP_OVERRIDES}" $i $3
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
        run_exp "jiant/config/taskmaster/base_roberta.conf" "${EXP_OVERRIDES}" $i $3
    done

}

#########################################
# Functions relating to main experiments
#########################################

function first_intermediate_exp() {
    # Initial intermediate task pretraining.
    # Usage: first_intermediate_task <intermediate_task_name> <config_number> <batch_size> <random_seed>
    OVERRIDES="exp_name=roberta-large, run_name=$1, batch_size=$3, reload_vocab=1"
    OVERRIDES+=", target_tasks=\"\", do_pretrain=1, do_target_task_training=0, input_module=roberta-large,pretrain_tasks=$1"
    run_exp "jiant/config/taskmaster/base_roberta.conf" "${OVERRIDES}" ${2} ${4}
}

function run_intermediate_to_target_task() {
    # Using a pretrained intermediate task, finetune on a target task.  ("STILTs" sheet)
    # This function can also be used to finetune on a probing task as well.
    # Usage: run_intermediate_to_target_task <intermediate_task> <target_task> <directory_to_project_dir> <config_number> <batch_size> <random_seed>
    OVERRIDES="exp_name=$1, run_name=$2"
    OVERRIDES+=", target_tasks=$2, load_model=1, load_target_train_checkpoint=$3/roberta-large/$1/model_*.best.th, pretrain_tasks=\"\""
    OVERRIDES+=", input_module=roberta-large, batch_size=$5, reload_vocab=1"
    OVERRIDES+=", do_pretrain=0, do_target_task_training=1"
    run_exp "jiant/config/taskmaster/base_roberta.conf" "${OVERRIDES}" ${4} ${5}
}

function run_intermediate_to_probing() {
    # Using a pretrained intermediate task, finetune on an probing task.  ("Probing" sheet)
    # Usage: run_intermediate_to_probing <intermediate_task> <probing task> <directory_to_project_dir> <config_number> <batch_size> <random_seed>
    OVERRIDES="exp_name=$1, run_name=$2"
    OVERRIDES+=", target_tasks=$2, load_model=1, load_target_train_checkpoint=$3/roberta-large/$1/$1/model_*.best.th, pretrain_tasks=\"\""
    OVERRIDES+=", input_module=roberta-large, batch_size=$5, reload_vocab=1"
    OVERRIDES+=", do_pretrain=0, do_target_task_training=1"
    run_exp "jiant/config/taskmaster/base_edgeprobe.conf" "${OVERRIDES}" ${4} ${6}
}

function run_intermediate_to_mixing() {
    # Using a pretrained intermediate task, use frozen encoder with mixing on an probing task.  ("Mixing" sheet)
    # Usage: run_intermediate_to_mixing <intermediate_task> <probing task> <directory_to_project_dir> <config_number> <batch_size> <random_seed>
    OVERRIDES="exp_name=$1, run_name=$2_mix"
    OVERRIDES+=", target_tasks=$2, load_model=1, load_target_train_checkpoint=$3/roberta-large/$1/model_*.best.th, pretrain_tasks=\"\""
    OVERRIDES+=", input_module=roberta-large, batch_size=$5, reload_vocab=1"
    OVERRIDES+=", transfer_paradigm=frozen, allow_untrained_encoder_parameters=1, pytorch_transformers_output_mode = mix"
    OVERRIDES+=", do_pretrain=0, do_target_task_training=1"
    run_exp "jiant/config/taskmaster/base_edgeprobe.conf" "${OVERRIDES}" ${4} ${6}
}
