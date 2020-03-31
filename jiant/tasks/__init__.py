# Import task definitions to register their tasks.
from jiant.tasks import (
    edge_probing,
    lm,
    nli_probing,
    qa,
    seq2seq,
    tasks,
    senteval_probing,
    acceptablity_probing,
)
from jiant.tasks.crosslingual import xnli

# REGISTRY needs to be available to modules within this package,
# but we also import it here to make it available at the package level.
from jiant.tasks.registry import REGISTRY

# Task class definition
from jiant.tasks.tasks import Task

##
# Task lists for handling as a group; these names correspond to the keys in
# the task registry.
ALL_GLUE_TASKS = [
    "sst",
    "cola",
    "mrpc",
    "qqp",
    "sts-b",
    "mnli",
    "qnli",
    "rte",
    "wnli",
    "glue-diagnostic",
]

ALL_SUPERGLUE_TASKS = [
    "boolq",
    "commitbank",
    "copa",
    "multirc",
    "record",
    "rte-superglue",
    "winograd-coreference",
    "wic",
    "broadcoverage-diagnostic",
    "winogender-diagnostic",
]

ALL_DIAGNOSTICS = ["broadcoverage-diagnostic", "winogender-diagnostic", "glue-diagnostic"]
# Tasks for the spring19_seminar; similar to cola but write predictions differently
ALL_COLA_NPI_TASKS = [
    "cola-npi-sup",
    "cola-npi-quessmp",
    "cola-npi-ques",
    "cola-npi-qnt",
    "cola-npi-only",
    "cola-npi-negsent",
    "cola-npi-negdet",
    "cola-npi-cond",
    "cola-npi-adv",
    "hd-cola-npi-sup",
    "hd-cola-npi-quessmp",
    "hd-cola-npi-ques",
    "hd-cola-npi-qnt",
    "hd-cola-npi-only",
    "hd-cola-npi-negsent",
    "hd-cola-npi-negdet",
    "hd-cola-npi-cond",
    "hd-cola-npi-adv",
    "all-cola-npi",
    "wilcox-npi",
    "npi-adv-li",
    "npi-adv-sc",
    "npi-adv-pr",
    "npi-cond-li",
    "npi-cond-sc",
    "npi-cond-pr",
    "npi-negdet-li",
    "npi-negdet-sc",
    "npi-negdet-pr",
    "npi-negsent-li",
    "npi-negsent-sc",
    "npi-negsent-pr",
    "npi-only-li",
    "npi-only-sc",
    "npi-only-pr",
    "npi-qnt-li",
    "npi-qnt-sc",
    "npi-qnt-pr",
    "npi-ques-li",
    "npi-ques-sc",
    "npi-ques-pr",
    "npi-quessmp-li",
    "npi-quessmp-sc",
    "npi-quessmp-pr",
    "npi-sup-li",
    "npi-sup-sc",
    "npi-sup-pr",
]

# Seq2seq tasks
ALL_SEQ2SEQ_TASKS = ["seg-wix"]

# people are mostly using nli-prob for now, but we will change to
# using individual tasks later, so better to have as a list
ALL_NLI_PROBING_TASKS = ["nli-prob", "nps", "nli-prob-prepswap", "nli-prob-negation", "nli-alt"]
