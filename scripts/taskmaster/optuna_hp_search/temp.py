import json

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
    "adversarial-nli",
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


def run_batch_size_check(batch_size):
    outputs = []
    for task_name in task_names:
        override = f'"reload_tasks=1, reload_vocab=1, do_pretrain=1, pretrain_tasks={task_name}, target_tasks={task_name}, run_name={task_name}, batch_size={batch_size}, max_epochs=1, val_interval=100, max_vals=1, patience=10000"'
        outputs.append(
            f'JIANT_CONF="jiant/config/taskmaster/clean_roberta.conf" JIANT_OVERRIDES={override} sbatch ~/jp100.sbatch'
        )
    return outputs


def run_optuna(parallel, gpus):
    outputs = []
    PROG="scripts/taskmaster/optuna_hp_search/run_trials" ARGS="winogrande 1 20" sbatch ~/p40.sbatch


outputs = run_batch_size_check(32)

with open("auto.sh", "w") as f:
    for line in outputs:
        f.write(line + "\n")
