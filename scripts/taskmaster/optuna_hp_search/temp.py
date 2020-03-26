import json
import math

task_names = [
    "sst",
    "socialiqa",
    "qqp",
    "mnli",
    "scitail",
    "qasrl",
    "qamr",
    "cosmosqa",
    "hellaswag",
    "commonsenseqa",
    "cola",
    "rte",
    "boolq",
    "commitbank",
    "copa",
    "multirc",
    "record",
    "wic",
    "ccg",
    "winograd-coreference",
    "winogrande",
    "edges-ner-ontonotes",
    "edges-srl-ontonotes",
    "edges-coref-ontonotes",
    "edges-spr1",
    "edges-spr2",
    "edges-dpr",
    "edges-rel-semeval",
    "edges-pos-ontonotes",
    "edges-nonterminal-ontonotes",
    "edges-dep-ud-ewt",
    "se-probing-word-content",
    "se-probing-tree-depth",
    "se-probing-top-constituents",
    "se-probing-bigram-shift",
    "se-probing-past-present",
    "se-probing-subj-number",
    "se-probing-obj-number",
    "se-probing-odd-man-out",
    "se-probing-coordination-inversion",
    "se-probing-sentence-length",
    "acceptability-wh",
    "acceptability-def",
    "acceptability-conj",
    "acceptability-eos",
]

with open("task_metadata.json", "r") as f:
    task_metadata = json.loads(f.read())


def run_batch_size_check(input_module):
    outputs = []

    for batch_size in [32, 16, 8, 4, 2]:
        for task_name in task_names:
            val_interval = min(5000, task_metadata[task_name]["training_size"] // batch_size)
            override = f'"do_pretrain=1, input_module={input_module}, pretrain_tasks={task_name}, target_tasks={task_name}, exp_name=bstest_{task_name}, run_name={task_name}_{batch_size}, batch_size={batch_size}, max_epochs=1, val_interval={val_interval}, delete_checkpoints_when_done=1, max_vals=1, patience=10000"'
            outputs.append(
                f'JIANT_CONF="jiant/config/taskmaster/clean_roberta.conf" JIANT_OVERRIDES={override} sbatch ~/jp40.sbatch'
            )
    return outputs


def run_optuna_trails(input_module):
    outputs = []
    for study_name, task in task_metadata.items():
        batch_size_limit = task[f'{input_module.split("-")[0]}_batch_size_limit']
        if batch_size_limit <= 4:
            gpu_available, sbatch = 4, "4p40.sbatch"
        elif batch_size_limit == 8:
            gpu_available, sbatch = 2, "2p40.sbatch"
        else:
            gpu_available, sbatch = 1, "p40.sbatch"

        if study_name.endswith("20k"):
            task_size = 20000
        elif study_name.endswith("5k"):
            task_size = 5000
        else:
            task_size = task["training_size"]

        if task_size >= 100000:
            parallel = 4
        elif task_size >= 20000:
            parallel = 2
        else:
            parallel = 1

        total_num_trails = 19 - task["optuna_trails"]
        for i in range(parallel):
            num_trails = (total_num_trails // parallel) + (i < (total_num_trails % parallel))
            if num_trails == 0:
                continue
            outputs.append(
                f'PROG="scripts/taskmaster/optuna_hp_search/run_trials" ARGS="{study_name} {gpu_available} {num_trails} {input_module}" sbatch ~/{sbatch}'
            )
    return outputs


def collect_optuna_trails(input_module):
    outputs = []
    for study_name, task in task_metadata.items():
        outputs.append(
            f'PROG="scripts/taskmaster/optuna_hp_search/collect_trials" ARGS="{study_name} {input_module}" sbatch ~/cpu.sbatch'
        )
    return outputs


def show_current_trail_count():
    outputs = []
    for task_name in task_names:
        outputs.append(f"tail optuna_{task_name}/results.tsv")
    return outputs


outputs = run_optuna_trails("albert-xxlarge-v2")

with open("auto.sh", "w") as f:
    for line in outputs:
        f.write(line + "\n")
