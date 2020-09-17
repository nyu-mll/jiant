# Directly download tasks when nlp format is different than original dataset
SQUAD_TASKS = {"squad_v1", "squad_v2"}
DIRECT_SUPERGLUE_TASKS_TO_DATA_URLS = {
    "wsc": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip",
    "multirc": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip",
    "record": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip",
}
OTHER_DOWNLOAD_TASKS = {"abductive_nli", "swag", "qamr", "qasrl"}
DIRECT_DOWNLOAD_TASKS = set(
    list(SQUAD_TASKS) + list(DIRECT_SUPERGLUE_TASKS_TO_DATA_URLS) + list(OTHER_DOWNLOAD_TASKS)
)
