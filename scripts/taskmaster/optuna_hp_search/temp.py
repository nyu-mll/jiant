import json

task_names = [
    # "sst",
    "socialiqa",
    "qqp",
    "mnli",
    # "scitail",
    # "qasrl",
    # "qamr",
    # "squad",
    "cosmosqa",
    "hellaswag",
    "commonsenseqa",
    # "cola",
    # "rte",
    "boolq",
    "commitbank",
    # "copa",
    "multirc",
    "record",
    # "wic",
    # "ccg",
    # "winograd-coreference",
    # "winogrande",
    "adversarial_nli",
    "mnli-snli-anli",
    # "edges-ner-ontonotes",
    # "edges-srl-ontonotes",
    # "edges-coref-ontonotes",
    # "edges-spr1",
    # "edges-spr2",
    # "edges-dpr",
    # "edges-rel-semeval",
    # "edges-pos-ontonotes",
    # "edges-nonterminal-ontonotes",
    # "edges-dep-ud-ewt",
    # "se-probing-word-content",
    # "se-probing-tree-depth",
    # "se-probing-top-constituents",
    # "se-probing-bigram-shift",
    # "se-probing-past-present",
    # "se-probing-subj-number",
    # "se-probing-obj-number",
    # "se-probing-odd-man-out",
    # "se-probing-coordination-inversion",
    # "se-probing-sentence-length",
    # "acceptability-wh",
    # "acceptability-def",
    # "acceptability-conj",
    # "acceptability-eos",
]

batch_size = 16
outputs = []

for task_name in task_names:
    override = f'"reload_tasks=1, reload_vocab=1, do_pretrain=1, pretrain_tasks={task_name}, run_name={task_name}, batch_size={batch_size}, max_epochs=1, patience=10000"'
    if "mnli-snli-anli" in task_name:
        override = override.replace("tasks=mnli-snli-anli", 'tasks=\\"mnli,snli,adversarial_nli\\"')
    outputs.append(
        f'JIANT_CONF="jiant/config/taskmaster/clean_roberta.conf" JIANT_OVERRIDES={override} sbatch ~/jp40.sbatch'
    )

task_metadata = [
    {"task_name": task_name, "batch_size_limit": 0, "training_size": 0} for task_name in task_names
]

# with open("task_metadata.json", "w") as f:
#     f.write(json.dumps(task_metadata))

with open("auto.sh", "w") as f:
    for line in outputs:
        f.write(line + "\n")
