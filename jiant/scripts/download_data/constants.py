# Directly download tasks when not available in HF Datasets, or HF Datasets version
#   is not suitable
SQUAD_TASKS = {"squad_v1", "squad_v2"}
DIRECT_SUPERGLUE_TASKS_TO_DATA_URLS = {
    "wsc": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip",
    "multirc": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip",
    "record": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip",
}

OTHER_DOWNLOAD_TASKS = {
    "abductive_nli",
    "fever_nli",
    "swag",
    "qamr",
    "qasrl",
    "newsqa",
    "mrqa_natural_questions",
    "piqa",
    "winogrande",
}

DIRECT_DOWNLOAD_TASKS = set(
    list(SQUAD_TASKS) + list(DIRECT_SUPERGLUE_TASKS_TO_DATA_URLS) + list(OTHER_DOWNLOAD_TASKS)
)
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
