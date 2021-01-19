# Requires variables:
#     MODEL_TYPE (e.g. xlm-roberta-large)
#     BASE_PATH
#
# Description:
#     This tokenizes and saves caches for all tasks (XTREME, MNLI, SQuAD)

### XNLI (uses MNLI for training)
for LG in ar bg de el en es fr hi ru sw th tr ur vi zh; do
    TASK=xnli_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 256 \
        --smart_truncate
done

### PAWS-X
TASK=pawsx_en
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val,test \
    --max_seq_length 256 \
    --smart_truncate
for LG in ar de es fr ja ko zh; do
    TASK=pawsx_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 256 \
        --smart_truncate
done

### UDPos
TASK=udpos_en
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val,test \
    --max_seq_length 256 \
    --smart_truncate
for LG in af ar bg de el es et eu fa fi fr he hi hu id it ja ko mr nl pt ru ta te tr ur vi zh; do
    TASK=udpos_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 256 \
        --smart_truncate
done
for LG in kk th tl yo; do
    TASK=udpos_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases test \
        --max_seq_length 256 \
        --smart_truncate
done

### PANX
TASK=panx_en
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val,test \
    --max_seq_length 256 \
    --smart_truncate
for LG in af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh; do
    TASK=panx_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 256 \
        --smart_truncate
done

### XQuAD (uses SQuAD for training)
for LG in ar de el en es hi ru th tr vi zh; do
    TASK=xquad_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val \
        --max_seq_length 384 \
        --smart_truncate
done

### MLQA (uses SQuAD for training)
for LG in ar de en es hi vi zh; do
    TASK=mlqa_${LG}_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 384 \
        --smart_truncate
done

### TyDiQA
TASK=tydiqa_en
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val \
    --max_seq_length 384 \
    --smart_truncate
for LG in ar bn fi id ko ru sw te; do
    TASK=tydiqa_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val \
        --max_seq_length 384 \
        --smart_truncate
done

### Bucc2018
for LG in de fr ru zh; do
    TASK=bucc2018_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 512 \
        --smart_truncate
done

### Tatoeba
for LG in af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr nl pt ru sw ta te th tl tr ur vi zh; do
    TASK=tatoeba_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val \
        --max_seq_length 512 \
        --smart_truncate
done

### MNLI and SQuAD
TASK=mnli
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val \
    --max_seq_length 256 \
    --smart_truncate
TASK=squad_v1
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val \
    --max_seq_length 384 \
    --smart_truncate
