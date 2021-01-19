# Requires variables:
#     MODEL_TYPE (e.g. xlm-roberta-large)
#     BASE_PATH
#
# Description:
#     This downloads a model (e.g. xlm-roberta-large)

python jiant/proj/main/export_model.py \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_base_path ${BASE_PATH}/models/${MODEL_TYPE}
