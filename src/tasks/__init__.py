# Import task definitions to register their tasks.
from . import tasks
from . import edge_probing
from . import lm
from . import mt
from . import nli_probing
from . import reddit

# REGISTRY needs to be available to modules within this package,
# but we also import it here to make it available at the package level.
from .registry import REGISTRY

# Task class definition
from .tasks import Task

##
# Task lists for handling as a group; these names correspond to the keys in
# the task registry.
ALL_GLUE_TASKS = ['sst', 'cola', 'mrpc', 'qqp', 'sts-b',
                  'mnli', 'qnli', 'rte', 'wnli', 'mnli-diagnostic']

ALL_COLA_NPI_TASKS = [
    'cola_npi_sup',
    'cola_npi_quessmp',
    'cola_npi_ques',
    'cola_npi_qnt',
    'cola_npi_only',
    'cola_npi_negsent',
    'cola_npi_negdet',
    'cola_npi_cond',
    'cola_npi_adv',
    'hd_cola_npi_sup',
    'hd_cola_npi_quessmp',
    'hd_cola_npi_ques',
    'hd_cola_npi_qnt',
    'hd_cola_npi_only',
    'hd_cola_npi_negsent',
    'hd_cola_npi_negdet',
    'hd_cola_npi_cond',
    'hd_cola_npi_adv',    
    'all_cola_npi'    
]

# people are mostly using nli-prob for now, but we will change to
# using individual tasks later, so better to have as a list
ALL_NLI_PROBING_TASKS = [
    'nli-prob',
    'nps',
    'nli-prob-prepswap',
    'nli-prob-negation',
    'nli-alt']

# Tasks for which we need to construct task-specific vocabularies
ALL_TARG_VOC_TASKS = ['wmt17_en_ru', 'wmt14_en_de',
                      'reddit_s2s', 'reddit_s2s_3.4G',
                      'wiki103_s2s', 'wiki2_s2s']
