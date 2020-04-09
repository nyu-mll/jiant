#!/bin/bash
# Functions to run Taskmaster tasks
# Usage: ./scripts/taskmaster/all_tasks.sh ${TASK} ${GPU_ID} ${SEED}
#   - TASK: one of the task-run names e.g. boolq
#   - GPU_ID: GPU to use, or -1 for CPU. Defaults to 0.
#   - SEED: random seed. Defaults to 111.

source user_config.sh
seed=${3:-111}
gpuid=${2:-0}

function boolq() {
    python main.py --config jiant/config/taskmaster/base_roberta.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = boolq, pretrain_tasks = \"boolq\", target_tasks = \"boolq\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 1000"
}

function commit() {
    python main.py --config jiant/config/taskmaster/base_roberta.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = commitbank, pretrain_tasks = \"commitbank\", target_tasks = \"commitbank\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 60"
}

function copa() {
    python main.py --config jiant/config/taskmaster/base_roberta.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = copa, pretrain_tasks = \"copa\", target_tasks = \"copa\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 100"
}

function multirc() {
    python main.py --config jiant/config/taskmaster/base_roberta.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = multirc, pretrain_tasks = \"multirc\", target_tasks = \"multirc\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 1000, val_data_limit = -1"
}

function record() {
    python main.py --config jiant/config/taskmaster/base_roberta.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = record, pretrain_tasks = \"record\", target_tasks = \"record\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 8, val_interval = 10000, val_data_limit = -1"
}

function rte() {
    python main.py --config jiant/config/taskmaster/base_roberta.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = rte, pretrain_tasks = \"rte-superglue\", target_tasks = \"rte-superglue,broadcoverage-diagnostic,winogender-diagnostic\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 625"
}

function wic() {
    python main.py --config jiant/config/taskmaster/base_roberta.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = wic, pretrain_tasks = \"wic\", target_tasks = \"wic\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 1000"
}

function wsc() {
    # NOTE: We use Adam b/c we were getting weird degenerate runs with BERT Adam
    python main.py --config jiant/config/taskmaster/base_roberta.conf --overrides "random_seed = ${seed}, cuda = ${gpuid}, run_name = wsc, pretrain_tasks = \"winograd-coreference\", target_tasks = \"winograd-coreference\", do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, batch_size = 4, val_interval = 139, optimizer = adam"
}

if [ $1 == "boolq" ]; then
    boolq
elif [ $1 == "commit" ]; then
    commit
elif [ $1 == "copa" ]; then
    copa
elif [ $1 == "multirc" ]; then
    multirc
elif [ $1 == "record" ]; then
    record
elif [ $1 == "rte" ]; then
    rte
elif [ $1 == "wic" ]; then
    wic
elif [ $1 == "wsc" ]; then
    wsc
fi
