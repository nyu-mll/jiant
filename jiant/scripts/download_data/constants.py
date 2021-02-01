# Directly download tasks when not available in HF Datasets, or HF Datasets version
#   is not suitable
SQUAD_TASKS = {"squad_v1", "squad_v2"}
DIRECT_SUPERGLUE_TASKS_TO_DATA_URLS = {
    "wsc": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip",
    "multirc": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip",
    "record": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip",
}

OTHER_DOWNLOAD_TASKS = {
    "abductive_nli",
    "arct",
    "fever_nli",
    "swag",
    "qamr",
    "qasrl",
    "newsqa",
    "mctaco",
    "mctest160",
    "mctest500",
    "mrqa_natural_questions",
    "mutual",
    "mutual_plus",
    "piqa",
    "winogrande",
    "ropes",
    "acceptability_definiteness",
    "acceptability_coord",
    "acceptability_eos",
    "acceptability_whwords",
    "senteval_bigram_shift",
    "senteval_coordination_inversion",
    "senteval_obj_number",
    "senteval_odd_man_out",
    "senteval_past_present",
    "senteval_sentence_length",
    "senteval_subj_number",
    "senteval_top_constituents",
    "senteval_tree_depth",
    "senteval_word_content",
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
    "race",
    "race_middle",
    "race_high",
    "quail",
}
