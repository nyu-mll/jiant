import os
from typing import Optional

from jiant.tasks.lib.abductive_nli import AbductiveNliTask
from jiant.tasks.lib.adversarial_nli import AdversarialNliTask
from jiant.tasks.lib.boolq import BoolQTask
from jiant.tasks.lib.ccg import CCGTask
from jiant.tasks.lib.cola import ColaTask
from jiant.tasks.lib.commitmentbank import CommitmentBankTask
from jiant.tasks.lib.commonsenseqa import CommonsenseQATask
from jiant.tasks.lib.copa import CopaTask
from jiant.tasks.lib.cosmosqa import CosmosQATask
from jiant.tasks.lib.hellaswag import HellaSwagTask
from jiant.tasks.lib.mlm_crosslingual_wiki import MLMCrosslingualWikiTask
from jiant.tasks.lib.mlm_wikitext_103 import MLMWikitext103Task
from jiant.tasks.lib.mnli import MnliTask
from jiant.tasks.lib.mrpc import MrpcTask
from jiant.tasks.lib.multirc import MultiRCTask
from jiant.tasks.lib.qqp import QqpTask
from jiant.tasks.lib.qnli import QnliTask
from jiant.tasks.lib.record import ReCoRDTask
from jiant.tasks.lib.rte import RteTask
from jiant.tasks.lib.scitail import SciTailTask
from jiant.tasks.lib.snli import SnliTask
from jiant.tasks.lib.socialiqa import SocialIQATask
from jiant.tasks.lib.squad import SquadTask
from jiant.tasks.lib.sst import SstTask
from jiant.tasks.lib.stsb import StsbTask
from jiant.tasks.lib.swag import SWAGTask
from jiant.tasks.lib.wic import WiCTask
from jiant.tasks.lib.wnli import WnliTask
from jiant.tasks.lib.wsc import WSCTask
from jiant.tasks.core import Task
from jiant.utils.python.io import read_json


TASK_DICT = {
    "abductive_nli": AbductiveNliTask,
    "adversarial_nli": AdversarialNliTask,
    "boolq": BoolQTask,
    "cb": CommitmentBankTask,
    "ccg": CCGTask,
    "cola": ColaTask,
    "commonsenseqa": CommonsenseQATask,
    "copa": CopaTask,
    "cosmosqa": CosmosQATask,
    "hellaswag": HellaSwagTask,
    "mlm_wikitext103": MLMWikitext103Task,
    "mlm_crosslingual_wiki": MLMCrosslingualWikiTask,
    "mlm": MLMWikitext103Task,
    "mnli": MnliTask,
    "mrc": MultiRCTask,
    "mrpc": MrpcTask,
    "qnli": QnliTask,
    "qqp": QqpTask,
    "record": ReCoRDTask,
    "rte": RteTask,
    "scitail": SciTailTask,
    "snli": SnliTask,
    "socialiqa": SocialIQATask,
    "squad": SquadTask,
    "sst": SstTask,
    "stsb": StsbTask,
    "swag": SWAGTask,
    "wic": WiCTask,
    "wnli": WnliTask,
    "wsc": WSCTask,
}


def get_task_class(task_name: str):
    task_class = TASK_DICT[task_name]
    assert issubclass(task_class, Task)
    return task_class


def create_task_from_config(config: dict, base_path: Optional[str] = None, verbose: bool = False):
    """Create task instance from task config.

    Args:
        config (Dict): task config map.
        base_path (str): if the path is not absolute, path is assumed to be relative to base_path.
        verbose (bool): True if task config should be printed during task creation.

    Returns:
        Task instance.

    """
    task_class = get_task_class(config["task"])
    for k in config["paths"].keys():
        path = config["paths"][k]
        # TODO: Refactor paths  (Issue #54)
        if isinstance(path, str) and not os.path.isabs(path):
            assert base_path
            config["paths"][k] = os.path.join(base_path, path)
    task_kwargs = config.get("kwargs", {})
    if verbose:
        print(task_class.__name__)
        for k, v in config["paths"].items():
            print(f"  [{k}]: {v}")
    # noinspection PyArgumentList
    return task_class(name=config["name"], path_dict=config["paths"], **task_kwargs)


def create_task_from_config_path(config_path: str, verbose: bool = False):
    """Creates task instance from task config filepath.

    Args:
        config_path (str): config filepath.
        verbose (bool): True if task config should be printed during task creation.

    Returns:
        Task instance.

    """
    return create_task_from_config(
        read_json(config_path), base_path=os.path.split(config_path)[0], verbose=verbose,
    )
