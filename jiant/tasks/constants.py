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
