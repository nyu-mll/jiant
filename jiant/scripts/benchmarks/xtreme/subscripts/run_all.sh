# Requires variables:
#     MODEL_TYPE (e.g. xlm-roberta-large)
#     BASE_PATH

bash a_download_model.sh
bash b_download_data.sh
bash c_tokenize_and_cache.sh
bash d_write_configs.sh
bash e_run_models.sh
