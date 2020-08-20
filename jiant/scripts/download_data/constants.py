# Directly download tasks when nlp format is different than original dataset
SQUAD_TASKS = {"squad_v1", "squad_v2"}
DIRECT_DOWNLOAD_TASKS_TO_DATA_URLS = {
    "wsc": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip",
    "multirc": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip",
    "record": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip",
}
DIRECT_DOWNLOAD_TASKS = DIRECT_DOWNLOAD_TASKS_TO_DATA_URLS.keys()
