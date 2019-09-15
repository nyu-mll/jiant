#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=40000
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
# We can choose target task from CoLA and all nine NPI tasks listed below:

#      - 'npi_sup'            NPI licensed by superlatives
#      - 'npi_quessmp'        NPI licensed by simple questions  
#      - 'npi_ques'           NPI licensed by questions
#      - 'npi_qnt'            NPI licensed by quantifiers
#      - 'npi_only'           NPI licensed by "only"
#      - 'npi_negsent'        NPI licensed by sentential negation
#      - 'npi_negdet'         NPI licensed by determiner negation
#      - 'npi_cond'           NPI licensed by conditionals
#      - 'npi_adv'            NPI licensed by adverbials

# These 9 different environments are to be found at https://github.com/alexwarstadt/data_generation/tree/master/outputs/npi/environments/splits
# Finetuning on one of the NPI task, we target all NPI tasks;
# Finetuning on eight of the NPI tasks, we target the remaining one;
# Finetuning on all the NPI tasks, we target all NPI tasks.

#### GET MODEL bertccg ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"\", pretrain_tasks = \"ccg\", do_pretrain = 1, do_full_eval = 0, do_target_task_training = 0" 

#### FINETUNE cola ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"cola\", pretrain_tasks = \"cola\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola,npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_cola
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_cola_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_cola
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_cola, target_tasks = \"cola,npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_cola/model_state_cola_best.th\", use_classifier=\"cola\"" 

#### FINETUNE npi_adv ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"npi_adv\", pretrain_tasks = \"npi_adv\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_adv
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_npi_adv_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_adv
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_npi_adv, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_adv/model_state_npi_adv_best.th\", use_classifier=\"npi_adv\"" 

#### FINETUNE npi_cond ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"npi_cond\", pretrain_tasks = \"npi_cond\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_cond
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_npi_cond_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_cond
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_npi_cond, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_cond/model_state_npi_cond_best.th\", use_classifier=\"npi_cond\"" 

#### FINETUNE npi_negdet ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"npi_negdet\", pretrain_tasks = \"npi_negdet\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_negdet
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_npi_negdet_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_negdet
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_npi_negdet, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_negdet/model_state_npi_negdet_best.th\", use_classifier=\"npi_negdet\"" 

#### FINETUNE npi_negsent ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"npi_negsent\", pretrain_tasks = \"npi_negsent\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_negsent
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_npi_negsent_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_negsent
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_npi_negsent, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_negsent/model_state_npi_negsent_best.th\", use_classifier=\"npi_negsent\"" 

#### FINETUNE npi_only ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"npi_only\", pretrain_tasks = \"npi_only\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_only
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_npi_only_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_only
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_npi_only, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_only/model_state_npi_only_best.th\", use_classifier=\"npi_only\"" 

#### FINETUNE npi_qnt ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"npi_qnt\", pretrain_tasks = \"npi_qnt\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_qnt
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_npi_qnt_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_qnt
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_npi_qnt, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_qnt/model_state_npi_qnt_best.th\", use_classifier=\"npi_qnt\"" 

#### FINETUNE npi_ques ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"npi_ques\", pretrain_tasks = \"npi_ques\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_ques
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_npi_ques_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_ques
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_npi_ques, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_ques/model_state_npi_ques_best.th\", use_classifier=\"npi_ques\"" 

#### FINETUNE npi_quessmp ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"npi_quessmp\", pretrain_tasks = \"npi_quessmp\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_quessmp
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_npi_quessmp_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_quessmp
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_npi_quessmp, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_quessmp/model_state_npi_quessmp_best.th\", use_classifier=\"npi_quessmp\"" 

#### FINETUNE hd_npi_sup ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"hd_npi_sup\", pretrain_tasks = \"hd_npi_sup\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,hd_npi_sup,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_sup
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_hd_npi_sup_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_sup
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_npi_sup, target_tasks = \"npi_sup,hd_npi_sup,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_sup/model_state_hd_npi_sup_best.th\", use_classifier=\"hd_npi_sup\"" 

#### FINETUNE npi_sup ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"npi_sup\", pretrain_tasks = \"npi_sup\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_sup
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_npi_sup_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_sup
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_npi_sup, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_npi_sup/model_state_npi_sup_best.th\", use_classifier=\"npi_sup\"" 

#### FINETUNE hd_npi_quessmp ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"hd_npi_quessmp\", pretrain_tasks = \"hd_npi_quessmp\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_quessmp,hd_npi_quessmp,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_quessmp
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_hd_npi_quessmp_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_quessmp
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_npi_quessmp, target_tasks = \"npi_quessmp,hd_npi_quessmp,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_quessmp/model_state_hd_npi_quessmp_best.th\", use_classifier=\"hd_npi_quessmp\"" 

#### FINETUNE hd_npi_ques ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"hd_npi_ques\", pretrain_tasks = \"hd_npi_ques\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_ques,hd_npi_ques,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_ques
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_hd_npi_ques_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_ques
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_npi_ques, target_tasks = \"npi_ques,hd_npi_ques,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_ques/model_state_hd_npi_ques_best.th\", use_classifier=\"hd_npi_ques\"" 

#### FINETUNE hd_npi_qnt ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"hd_npi_qnt\", pretrain_tasks = \"hd_npi_qnt\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_qnt,hd_npi_qnt,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_qnt
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_hd_npi_qnt_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_qnt
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_npi_qnt, target_tasks = \"npi_qnt,hd_npi_qnt,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_qnt/model_state_hd_npi_qnt_best.th\", use_classifier=\"hd_npi_qnt\"" 

#### FINETUNE hd_npi_only ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"hd_npi_only\", pretrain_tasks = \"hd_npi_only\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_only,hd_npi_only,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_only
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_hd_npi_only_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_only
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_npi_only, target_tasks = \"npi_only,hd_npi_only,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_only/model_state_hd_npi_only_best.th\", use_classifier=\"hd_npi_only\"" 

#### FINETUNE hd_npi_negsent ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"hd_npi_negsent\", pretrain_tasks = \"hd_npi_negsent\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_negsent,hd_npi_negsent,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_negsent
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_hd_npi_negsent_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_negsent
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_npi_negsent, target_tasks = \"npi_negsent,hd_npi_negsent,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_negsent/model_state_hd_npi_negsent_best.th\", use_classifier=\"hd_npi_negsent\"" 

#### FINETUNE hd_npi_negdet ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"hd_npi_negdet\", pretrain_tasks = \"hd_npi_negdet\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_negdet,hd_npi_negdet,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_negdet
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_hd_npi_negdet_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_negdet
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_npi_negdet, target_tasks = \"npi_negdet,hd_npi_negdet,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_negdet/model_state_hd_npi_negdet_best.th\", use_classifier=\"hd_npi_negdet\"" 

#### FINETUNE hd_npi_cond ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"hd_npi_cond\", pretrain_tasks = \"hd_npi_cond\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_cond,hd_npi_cond,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_cond
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_hd_npi_cond_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_cond
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_npi_cond, target_tasks = \"npi_cond,hd_npi_cond,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_cond/model_state_hd_npi_cond_best.th\", use_classifier=\"hd_npi_cond\"" 

#### FINETUNE hd_npi_adv ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"hd_npi_adv\", pretrain_tasks = \"hd_npi_adv\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_adv,hd_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_adv
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_hd_npi_adv_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_adv
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_hd_npi_adv, target_tasks = \"npi_adv,hd_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_hd_npi_adv/model_state_hd_npi_adv_best.th\", use_classifier=\"hd_npi_adv\"" 

#### FINETUNE all_npi ####
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_model, target_tasks = \"all_npi\", pretrain_tasks = \"all_npi\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,all_npi,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_all_npi
mv $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_model/model_state_all_npi_best.th $JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_all_npi
python main.py --config_file jiant/config/bert_npi/bert.conf \
    --overrides "exp_name = npi_bertccg, run_name = run_bertccg_all_npi, target_tasks = \"npi_sup,npi_quessmp,npi_ques,npi_qnt,npi_only,npi_negsent,npi_negdet,npi_cond,npi_adv,all_npi,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertccg/run_bertccg_all_npi/model_state_all_npi_best.th\", use_classifier=\"all_npi\"" 

