"""Use this for tasks that can be obtained from NLP without further/special processing"""

import jiant.scripts.download_data.utils as download_utils
import jiant.utils.python.io as py_io

# Note to future selves: beware of circular imports when refactoring
from jiant.tasks.retrieval import (
    ColaTask,
    MrpcTask,
    QnliTask,
    QqpTask,
    RteTask,
    SstTask,
    WnliTask,
    BoolQTask,
    CommitmentBankTask,
    WiCTask,
    WSCTask,
    SuperglueWinogenderDiagnosticsTask,
    GlueDiagnosticsTask,
    # === Additional for IRT === #
    WinograndeTask,
)


NLP_CONVERSION_DICT = {
    # === GLUE === #
    "cola": {
        "path": "glue",
        "name": "cola",
        "field_map": {"sentence": "text"},
        "label_map": ColaTask.ID_TO_LABEL,
    },
    "mnli": {
        "path": "glue",
        "name": "mnli",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "phase_map": {"validation_matched": "val", "test_matched": "test"},
        "phase_list": ["train", "val", "test"],
    },
    "mnli_mismatched": {
        "path": "glue",
        "name": "mnli",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "phase_map": {"validation_mismatched": "val", "test_mismatched": "test"},
        "phase_list": ["val", "test"],
        "jiant_task_name": "mnli_mismatched",
    },
    "mrpc": {
        "path": "glue",
        "name": "mrpc",
        "field_map": {"sentence1": "text_a", "sentence2": "text_b"},
        "label_map": MrpcTask.ID_TO_LABEL,
    },
    "qnli": {
        "path": "glue",
        "name": "qnli",
        "field_map": {"question": "premise", "sentence": "hypothesis"},
        "label_map": QnliTask.ID_TO_LABEL,
    },
    "qqp": {
        "path": "glue",
        "name": "qqp",
        "field_map": {"question1": "text_a", "question2": "text_b"},
        "label_map": QqpTask.ID_TO_LABEL,
    },
    "rte": {
        "path": "glue",
        "name": "rte",
        "field_map": {"sentence1": "premise", "sentence2": "hypothesis"},
        "label_map": RteTask.ID_TO_LABEL,
    },
    "sst": {
        "path": "glue",
        "name": "sst2",
        "field_map": {"sentence": "text"},
        "label_map": SstTask.ID_TO_LABEL,
    },
    "stsb": {
        "path": "glue",
        "name": "stsb",
        "field_map": {"sentence1": "text_a", "sentence2": "text_b"},
    },
    "wnli": {
        "path": "glue",
        "name": "wnli",
        "field_map": {"sentence1": "premise", "sentence2": "hypothesis"},
        "label_map": WnliTask.ID_TO_LABEL,
    },
    "glue_diagnostics": {
        "path": "glue",
        "name": "ax",
        "label_map": GlueDiagnosticsTask.ID_TO_LABEL,
        "phase_map": None,
        "jiant_task_name": "glue_diagnostics",
    },
    # === SuperGLUE === #
    "boolq": {"path": "super_glue", "name": "boolq", "label_map": BoolQTask.ID_TO_LABEL},
    "cb": {"path": "super_glue", "name": "cb", "label_map": CommitmentBankTask.ID_TO_LABEL},
    "copa": {"path": "super_glue", "name": "copa"},
    "multirc": {"path": "super_glue", "name": "multirc"},
    "record": {"path": "super_glue", "name": "record"},
    "wic": {"path": "super_glue", "name": "wic", "label_map": WiCTask.ID_TO_LABEL},
    "wsc": {"path": "super_glue", "name": "wsc.fixed", "label_map": WSCTask.ID_TO_LABEL},
    "superglue_broadcoverage_diagnostics": {
        "path": "super_glue",
        "name": "axb",
        "field_map": {"sentence1": "premise", "sentence2": "hypothesis"},
        "label_map": RteTask.ID_TO_LABEL,
        "phase_map": None,
        "jiant_task_name": "rte",
    },
    "superglue_winogender_diagnostics": {
        "path": "super_glue",
        "name": "axg",
        "label_map": SuperglueWinogenderDiagnosticsTask.ID_TO_LABEL,
        "phase_map": None,
        "jiant_task_name": "superglue_axg",
    },
    # === Other === #
    "snli": {"path": "snli", "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"}},
    "commonsenseqa": {"path": "commonsense_qa", "phase_list": ["train", "val", "test"]},
    "hellaswag": {
        "path": "hellaswag",
        "phase_list": ["train", "val", "test"],
        "label_map": {"0": 0, "1": 1, "2": 2, "3": 3},
    },
    "cosmosqa": {"path": "cosmos_qa", "phase_list": ["train", "val", "test"]},
    "socialiqa": {"path": "social_i_qa", "phase_list": ["train", "val"]},
    "scitail": {"path": "scitail", "name": "tsv_format", "phase_list": ["train", "val", "test"]},
    # === Additional for IRT === #
    "winogrande":{
        "path":"winogrande",
        "name":"winogrande_xs",
        "phase_map": {"dev": "val"},
        "phase_list": ["train", "val", "test"],
        "label_map":WinograndeTask.ID_TO_CHOICE,
    }
}

# NLP uses "validation", we use "val"
DEFAULT_PHASE_MAP = {"validation": "val"}


def download_data_and_write_config(task_name: str, task_data_path: str, task_config_path: str):
    nlp_conversion_metadata = NLP_CONVERSION_DICT[task_name]
    examples_dict = download_utils.convert_nlp_dataset_to_examples(
        path=nlp_conversion_metadata["path"],
        name=nlp_conversion_metadata.get("name"),
        field_map=nlp_conversion_metadata.get("field_map"),
        label_map=nlp_conversion_metadata.get("label_map"),
        phase_map=nlp_conversion_metadata.get("phase_map", DEFAULT_PHASE_MAP),
        phase_list=nlp_conversion_metadata.get("phase_list"),
    )
    paths_dict = download_utils.write_examples_to_jsonls(
        examples_dict=examples_dict, task_data_path=task_data_path,
    )
    jiant_task_name = nlp_conversion_metadata.get("jiant_task_name", task_name)
    py_io.write_json(
        data={"task": jiant_task_name, "paths": paths_dict, "name": task_name},
        path=task_config_path,
    )
