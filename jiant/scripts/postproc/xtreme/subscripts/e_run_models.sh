# Requires variables:
#     MODEL_TYPE (e.g. xlm-roberta-large)
#     BASE_PATH
#
# Description:
#     This runs models for both fine-tuned and retrieval XTREME tasks
#     Ideally, this should be run in parallel on a cluster.

for TASK in xnli pawsx udpos panx xquad mlqa tydiqa; do
    python jiant/proj/main/runscript.py \
        run_with_continue \
        --ZZsrc ${BASE_PATH}/models/${MODEL_TYPE}/config.json \
        --jiant_task_container_config_path ${BASE_PATH}/runconfigs/${TASK}.json \
        --model_load_mode from_transformers \
        --learning_rate 1e-5 \
        --eval_every_steps 1000 \
        --no_improvements_for_n_evals 30 \
        --do_save \
        --force_overwrite \
        --do_train --do_val \
        --output_dir ${BASE_PATH}/runs/${TASK}
done

for TASK in bucc2018 tatoeba; do
    python jiant/proj/main/runscript.py \
        run_with_continue \
        --ZZsrc ${BASE_PATH}/models/${MODEL_TYPE}/config.json \
        --jiant_task_container_config_path ${BASE_PATH}/runconfigs/${TASK}.json \
        --model_load_mode from_transformers \
        --force_overwrite \
        --do_val \
        --output_dir ${BASE_PATH}/runs/${TASK}
done
