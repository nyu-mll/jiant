# Requires variables:
#     MODEL_TYPE (e.g. xlm-roberta-large)
#
# Description:
#     This downloads the XTREME datasets, as well as MNLI and SQuAD for training


python jiant/scripts/download_data/runscript.py \
    download \
    --benchmark XTREME \
    --output_path ${BASE_PATH}/tasks/
python jiant/scripts/download_data/runscript.py \
    download \
    --tasks mnli squad_v1 \
    --output_path ${BASE_PATH}/tasks/
