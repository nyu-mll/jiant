# Requires variables:
#     MODEL_TYPE (e.g. xlm-roberta-large)
#     BASE_PATH
#
# Description:
#     This write run-configs for each of the XTREME tasks

mkdir -p ${BASE_PATH}/runconfigs

# XNLI
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task xnli \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 2 --train_batch_size 4 --gradient_accumulation_steps 8 \
    --output_path ${BASE_PATH}/runconfigs/xnli.json

# PAWS-X
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task pawsx \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 5 --train_batch_size 4 --gradient_accumulation_steps 8 \
    --output_path ${BASE_PATH}/runconfigs/pawsx.json

# UDPOS
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task udpos \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 10 --train_batch_size 4 --gradient_accumulation_steps 8 \
    --output_path ${BASE_PATH}/runconfigs/udpos.json

# PANX
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task panx \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 10 --train_batch_size 4 --gradient_accumulation_steps 8 \
    --output_path ${BASE_PATH}/runconfigs/panx.json

# XQuAD
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task xquad \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 2 --train_batch_size 4 --gradient_accumulation_steps 4 \
    --output_path ${BASE_PATH}/runconfigs/xquad.json

# MLQA
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task mlqa \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 2 --train_batch_size 4 --gradient_accumulation_steps 4 \
    --output_path ${BASE_PATH}/runconfigs/mlqa.json

# TyDiQA
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task tydiqa \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 2 --train_batch_size 4 --gradient_accumulation_steps 4 \
    --output_path ${BASE_PATH}/runconfigs/tydiqa.json

# Bucc2018
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task bucc2018 \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --output_path ${BASE_PATH}/runconfigs/bucc2018.json

# Tatoeba
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task tatoeba \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --output_path ${BASE_PATH}/runconfigs/tatoeba.json