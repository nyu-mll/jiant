# Import task definitions to register their tasks.
from jiant.tasks import edge_probing, lm, nli_probing, qa, seq2seq, tasks

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
    "cola_npi_sup",
    "cola_npi_quessmp",
    "cola_npi_ques",
    "cola_npi_qnt",
    "cola_npi_only",
    "cola_npi_negsent",
    "cola_npi_negdet",
    "cola_npi_cond",
    "cola_npi_adv",
    "hd_cola_npi_sup",
    "hd_cola_npi_quessmp",
    "hd_cola_npi_ques",
    "hd_cola_npi_qnt",
    "hd_cola_npi_only",
    "hd_cola_npi_negsent",
    "hd_cola_npi_negdet",
    "hd_cola_npi_cond",
    "hd_cola_npi_adv",
    "all_cola_npi",
    "wilcox_npi",
    "npi_adv_li",
    "npi_adv_sc",
    "npi_adv_pr",
    "npi_cond_li",
    "npi_cond_sc",
    "npi_cond_pr",
    "npi_negdet_li",
    "npi_negdet_sc",
    "npi_negdet_pr",
    "npi_negsent_li",
    "npi_negsent_sc",
    "npi_negsent_pr",
    "npi_only_li",
    "npi_only_sc",
    "npi_only_pr",
    "npi_qnt_li",
    "npi_qnt_sc",
    "npi_qnt_pr",
    "npi_ques_li",
    "npi_ques_sc",
    "npi_ques_pr",
    "npi_quessmp_li",
    "npi_quessmp_sc",
    "npi_quessmp_pr",
    "npi_sup_li",
    "npi_sup_sc",
    "npi_sup_pr",
]

# Seq2seq tasks
ALL_SEQ2SEQ_TASKS = ["seg_wix"]

# people are mostly using nli-prob for now, but we will change to
# using individual tasks later, so better to have as a list
ALL_NLI_PROBING_TASKS = ["nli-prob", "nps", "nli-prob-prepswap", "nli-prob-negation", "nli-alt"]
