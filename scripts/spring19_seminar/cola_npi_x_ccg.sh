#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --job-name=myrun
#SBATCH --output=slurm_%j.out

# Naming Scheme
# the experiment name is named after "npi_[MODEL NAME]", e.g., npi_bert
# the run name falls into three kinds
# (1) if the run goes through pretraining (e.g., ccg, mnli, or none) and some finetuning tasks and finally the target tasks,
#     then the run name is "run_[MODEL NAME][PRETRAIN]_[FINETUNE NAME]";
# (2) if the run merely does pretraining to generate a best model for loading in later runs,
#     then the run name is "run_[MODEL NAME][PRETRAIN]_model";
# (3) if the run loads a pretrained model and goes through finetuning tasks and then target tasks,
#     then the run name is "run_[MODEL_NAME][PRETRAIN]_[FINTUNE NAME]", same as in (1).


# Finetune-Target Tasks Combinations
# All the finetune and target tasks are CoLA-like.
# Finetuning on CoLA, we target CoLA and all nine NPI tasks listed below:

#      - 'cola_npi_sup'            NPI licensed by superlatives
#      - 'cola_npi_quessmp'        NPI licensed by simple questions  
#      - 'cola_npi_ques'           NPI licensed by questions
#      - 'cola_npi_qnt'            NPI licensed by quantifiers
#      - 'cola_npi_only'           NPI licensed by "only"
#      - 'cola_npi_negsent'        NPI licensed by sentential negation
#      - 'cola_npi_negdet'         NPI licensed by determiner negation
#      - 'cola_npi_cond'           NPI licensed by conditionals
#      - 'cola_npi_adv'            NPI licensed by adverbials

# These 9 different environments are to be found at https://github.com/alexwarstadt/data_generation/tree/master/outputs/npi/environments/splits
# Finetuning on one of the NPI task, we target all NPI tasks;
# Finetuning on eight of the NPI tasks, we target the remaining one.


python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_model, target_tasks = \"\", pretrain_tasks = \"ccg\", do_pretrain = 1, do_full_eval = 0" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola, target_tasks = \"cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola_npi_adv
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_adv, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola_npi_cond
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_cond, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_cond\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_negdet, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_negdet\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_negsent, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_negsent\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola_npi_only
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_only, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_only\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_qnt, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_qnt\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola_npi_ques
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_ques, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_ques\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_quessmp, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_quessmp\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_hd_cola_npi_sup
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_sup, target_tasks = \"cola_npi_sup\", pretrain_tasks = \"cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_cola_npi_sup
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_cola_npi_sup, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_sup\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_hd_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_quessmp, target_tasks = \"cola_npi_quessmp\", pretrain_tasks = \"cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_hd_cola_npi_ques
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_ques, target_tasks = \"cola_npi_ques\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_hd_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_qnt, target_tasks = \"cola_npi_qnt\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_hd_cola_npi_only
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_only, target_tasks = \"cola_npi_only\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_hd_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_negsent, target_tasks = \"cola_npi_negsent\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_hd_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_negdet, target_tasks = \"cola_npi_negdet\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_hd_cola_npi_cond
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_cond, target_tasks = \"cola_npi_cond\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_hd_cola_npi_adv
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_hd_cola_npi_adv, target_tasks = \"cola_npi_adv\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertccg_allnpi
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertccg_allnpi, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_model, target_tasks = \"\", pretrain_tasks = \"ccg\", do_pretrain = 1, do_full_eval = 0" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola, target_tasks = \"cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola_npi_adv
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_adv, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola_npi_cond
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_cond, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_cond\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_negdet, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_negdet\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_negsent, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_negsent\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola_npi_only
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_only, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_only\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_qnt, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_qnt\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola_npi_ques
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_ques, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_ques\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_quessmp, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_quessmp\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_hd_cola_npi_sup
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_sup, target_tasks = \"cola_npi_sup\", pretrain_tasks = \"cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_cola_npi_sup
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_cola_npi_sup, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_sup\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_hd_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_quessmp, target_tasks = \"cola_npi_quessmp\", pretrain_tasks = \"cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_hd_cola_npi_ques
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_ques, target_tasks = \"cola_npi_ques\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_hd_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_qnt, target_tasks = \"cola_npi_qnt\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_hd_cola_npi_only
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_only, target_tasks = \"cola_npi_only\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_hd_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_negsent, target_tasks = \"cola_npi_negsent\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_hd_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_negdet, target_tasks = \"cola_npi_negdet\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_hd_cola_npi_cond
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_cond, target_tasks = \"cola_npi_cond\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_hd_cola_npi_adv
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_hd_cola_npi_adv, target_tasks = \"cola_npi_adv\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_gloveccg_allnpi
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_gloveccg_allnpi, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

