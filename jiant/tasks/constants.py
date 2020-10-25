GLUE_TASKS = {
    "cola",
    "sst",
    "mrpc",
    "qqp",
    "stsb",
    "mnli",
    "mnli_mismatched",
    "qnli",
    "rte",
    "wnli",
    "glue_diagnostics",
}

SUPERGLUE_TASKS = {
    "cb",
    "copa",
    "multirc",
    "wic",
    "wsc",
    "boolq",
    "record",
    "rte",
    "superglue_broadcoverage_diagnostics",
    "superglue_winogender_diagnostics",
}

OTHER_HF_DATASETS_TASKS = {
    "snli",
    "commonsenseqa",
    "hellaswag",
    "cosmosqa",
    "socialiqa",
    "scitail",
    "quoref",
    "adversarial_nli_r1",
    "adversarial_nli_r2",
    "adversarial_nli_r3",
    "arc_easy",
    "arc_challenge",
}

XTREME_TASKS = {
    "xnli",
    "pawsx",
    "udpos",
    "panx",
    "xquad",
    "mlqa",
    "tydiqa",
    "bucc2018",
    "tatoeba",
}

BENCHMARKS = {"GLUE": GLUE_TASKS, "SUPERGLUE": SUPERGLUE_TASKS, "XTREME": XTREME_TASKS}
