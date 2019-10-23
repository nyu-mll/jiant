# intermediate_trian this_one 
# target_train intermediate_train target_task


function run_exp() {
    # Helper function to invoke main.py.
    # Don't run this directly - use the experiment functions below,
    # or create a new one for a new experiment suite.
    # Usage: run_exp <config_file> <overrides>
    CONFIG_FILE=$1
    OVERRIDES=$2
    declare -a args
    args+=( --config_file "${CONFIG_FILE}" )
    args+=( -o "${OVERRIDES}" --remote_log )
    python main.py "${args[@]}"
}

function first_intermediate_exp() {
    # Initial intermdiate task pretraining.
    # Usage: first_intermediate_task <intermediate_task_name>
    OVERRIDES="exp_name=robert-large, run_name=$1"
    OVERRIDES+=", target_tasks=$1, do_pretrain=0, do_target_task_training=1, input_module=roberta-large"
    run_exp "jiant/config/taskmaster/base_roberta.conf" "${OVERRIDES}"
}

function run_intermediate_to_target_task() {
    # Using a pretrained intermediate task, finetune on a target task. 
    # This function can also be used to finetune on a probing task as well. 
    # Usage: run_intermediate_to_target_task <intemeidate_task> <target_task> <directory_to_jiant>
    OVERRIDES="exp_name=$1, run_name=$2"
    OVERRIDES+=", target_tasks=$2, load_model=1, load_target_train_checkpoint=$3/roberta-large/$1,"
    OVERRIDES+="input_module=roberta-large"
    OVERRIDES+="do_pretrain=0, do_target_task_training=1, "
    run_exp "jiant/config/taskmaster/base_roberta.conf" "${OVERRIDES}"
}

