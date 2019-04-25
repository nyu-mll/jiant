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

# Tasks for the spring19_seminar; similar to cola but write predictions differently
ALL_NPI_TASKS = ["npi_adv", "npi_cond", "npi_negdet", "npi_negsent", "npi_only", "npi_ques", "npi_quessmp", "npi_qnt", "npi_sup", 
                "npi_adv_li", "npi_adv_sc", "npi_adv_pr", 
                "npi_cond_li", "npi_cond_sc", "npi_cond_pr", 
                "npi_negdet_li", "npi_negdet_sc", "npi_negdet_pr", 
                "npi_negsent_li", "npi_negsent_sc", "npi_negsent_pr", 
                "npi_only_li", "npi_only_sc", "npi_only_pr", 
                "npi_qnt_li", "npi_qnt_sc", "npi_qnt_pr", 
                "npi_ques_li", "npi_ques_sc", "npi_ques_pr", 
                "npi_quessmp_li", "npi_quessmp_sc", "npi_quessmp_pr", 
                "npi_sup_li", "npi_sup_sc", "npi_sup_pr"]

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
