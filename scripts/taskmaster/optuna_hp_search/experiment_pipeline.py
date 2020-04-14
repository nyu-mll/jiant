from generate_scripts import (
    preprocess_tasks,
    run_exp_init,
    run_batch_size_check,
    update_batch_size_check,
    run_main_optuna_trials,
    run_additional_optuna_trials,
    run_pretrain,
    run_target_train,
    write_script_file,
)

# Step 0.a
# preprocess edgeprobing, ccg tasks
write_script_file("preprocess_roberta.sh", preprocess_tasks("roberta-large"))
write_script_file("preprocess_albert.sh", preprocess_tasks("albert-xxlarge-v2"))

# Step 0.b
# create exps, this allow us to avoid reload_vocab related bugs.
write_script_file("create_exps_roberta.sh", run_exp_init("roberta-large"))
write_script_file("create_exps_albert.sh", run_exp_init("albert-xxlarge-v2"))


# step 1.a
# check greatest batch size
write_script_file("batch_size_roberta.sh", run_batch_size_check("roberta-large"))
write_script_file("batch_size_albert.sh", run_batch_size_check("albert-xxlarge-v2"))


# # step 1.b
# # update metadata with greatest batch size
# update_batch_size_check("roberta-large")
# update_batch_size_check("albert-xxlarge-v2")


# # step 2.a
# # main hp search
# write_script_file("main_hp_search_roberta.sh", run_main_optuna_trials("roberta-large"))
# write_script_file("main_hp_search_albert.sh", run_main_optuna_trials("albert-xxlarge-v2"))

# # step 2.b
# # finalize hp search
# write_script_file("finalize_hp_search_roberta.sh", run_additional_optuna_trials("roberta-large"))
# write_script_file("finalize_hp_search_albert.sh", run_additional_optuna_trials("albert-xxlarge-v2"))


# # step 3
# # interm training, w/ w/o MLM
# commands, roberta_checkpoints = run_pretrain("roberta-large")
# write_script_file("pretrain_roberta.sh", commands)
# commands, albert_checkpoints = run_pretrain("albert-xxlarge-v2")
# write_script_file("pretrain_albert.sh", commands)


# # step 4
# # finetune target & probing & finite size probing
# write_script_file("pretrain_roberta.sh", run_target_train("roberta-large", roberta_checkpoints))
# write_script_file("pretrain_albert.sh", run_target_train("albert-xxlarge-v2", albert_checkpoints))
