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
# Finetuning on eight of the NPI tasks, we target the remaining one;
# Finetuning on all the NPI tasks, we target all NPI tasks.

#### GET MODEL bertmnli ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"\", pretrain_tasks = \"mnli\", do_pretrain = 1, do_full_eval = 0, do_target_task_training = 0" 

#### FINETUNE cola ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola\", pretrain_tasks = \"cola\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola, target_tasks = \"cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola/model_state_cola_best.th\", use_classifier=\"cola\"" 

#### FINETUNE cola_npi_adv ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola_npi_adv\", pretrain_tasks = \"cola_npi_adv\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_adv
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_npi_adv_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_adv
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_adv, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_adv/model_state_cola_npi_adv_best.th\", use_classifier=\"cola_npi_adv\"" 

#### FINETUNE cola_npi_cond ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola_npi_cond\", pretrain_tasks = \"cola_npi_cond\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_cond
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_npi_cond_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_cond
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_cond, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_cond/model_state_cola_npi_cond_best.th\", use_classifier=\"cola_npi_cond\"" 

#### FINETUNE cola_npi_negdet ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola_npi_negdet\", pretrain_tasks = \"cola_npi_negdet\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_negdet
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_npi_negdet_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_negdet, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_negdet/model_state_cola_npi_negdet_best.th\", use_classifier=\"cola_npi_negdet\"" 

#### FINETUNE cola_npi_negsent ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola_npi_negsent\", pretrain_tasks = \"cola_npi_negsent\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_negsent
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_npi_negsent_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_negsent, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_negsent/model_state_cola_npi_negsent_best.th\", use_classifier=\"cola_npi_negsent\"" 

#### FINETUNE cola_npi_only ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola_npi_only\", pretrain_tasks = \"cola_npi_only\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_only
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_npi_only_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_only
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_only, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_only/model_state_cola_npi_only_best.th\", use_classifier=\"cola_npi_only\"" 

#### FINETUNE cola_npi_qnt ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola_npi_qnt\", pretrain_tasks = \"cola_npi_qnt\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_qnt
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_npi_qnt_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_qnt, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_qnt/model_state_cola_npi_qnt_best.th\", use_classifier=\"cola_npi_qnt\"" 

#### FINETUNE cola_npi_ques ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola_npi_ques\", pretrain_tasks = \"cola_npi_ques\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_ques
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_npi_ques_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_ques
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_ques, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_ques/model_state_cola_npi_ques_best.th\", use_classifier=\"cola_npi_ques\"" 

#### FINETUNE cola_npi_quessmp ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola_npi_quessmp\", pretrain_tasks = \"cola_npi_quessmp\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_quessmp
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_npi_quessmp_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_quessmp, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_quessmp/model_state_cola_npi_quessmp_best.th\", use_classifier=\"cola_npi_quessmp\"" 

#### FINETUNE hd_cola_npi_sup ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"hd_cola_npi_sup\", pretrain_tasks = \"hd_cola_npi_sup\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,hd_cola_npi_sup,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_sup
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_sup_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_sup
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_sup, target_tasks = \"cola_npi_sup,hd_cola_npi_sup,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_sup/model_state_hd_cola_npi_sup_best.th\", use_classifier=\"hd_cola_npi_sup\"" 

#### FINETUNE cola_npi_sup ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"cola_npi_sup\", pretrain_tasks = \"cola_npi_sup\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_sup
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_cola_npi_sup_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_sup
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_cola_npi_sup, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_cola_npi_sup/model_state_cola_npi_sup_best.th\", use_classifier=\"cola_npi_sup\"" 

#### FINETUNE hd_cola_npi_quessmp ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"hd_cola_npi_quessmp\", pretrain_tasks = \"hd_cola_npi_quessmp\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_quessmp,hd_cola_npi_quessmp,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_quessmp
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_quessmp_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_quessmp, target_tasks = \"cola_npi_quessmp,hd_cola_npi_quessmp,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_quessmp/model_state_hd_cola_npi_quessmp_best.th\", use_classifier=\"hd_cola_npi_quessmp\"" 

#### FINETUNE hd_cola_npi_ques ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"hd_cola_npi_ques\", pretrain_tasks = \"hd_cola_npi_ques\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_ques,hd_cola_npi_ques,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_ques
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_ques_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_ques
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_ques, target_tasks = \"cola_npi_ques,hd_cola_npi_ques,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_ques/model_state_hd_cola_npi_ques_best.th\", use_classifier=\"hd_cola_npi_ques\"" 

#### FINETUNE hd_cola_npi_qnt ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"hd_cola_npi_qnt\", pretrain_tasks = \"hd_cola_npi_qnt\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_qnt,hd_cola_npi_qnt,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_qnt
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_qnt_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_qnt, target_tasks = \"cola_npi_qnt,hd_cola_npi_qnt,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_qnt/model_state_hd_cola_npi_qnt_best.th\", use_classifier=\"hd_cola_npi_qnt\"" 

#### FINETUNE hd_cola_npi_only ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"hd_cola_npi_only\", pretrain_tasks = \"hd_cola_npi_only\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_only,hd_cola_npi_only,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_only
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_only_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_only
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_only, target_tasks = \"cola_npi_only,hd_cola_npi_only,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_only/model_state_hd_cola_npi_only_best.th\", use_classifier=\"hd_cola_npi_only\"" 

#### FINETUNE hd_cola_npi_negsent ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"hd_cola_npi_negsent\", pretrain_tasks = \"hd_cola_npi_negsent\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_negsent,hd_cola_npi_negsent,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_negsent
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_negsent_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_negsent, target_tasks = \"cola_npi_negsent,hd_cola_npi_negsent,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_negsent/model_state_hd_cola_npi_negsent_best.th\", use_classifier=\"hd_cola_npi_negsent\"" 

#### FINETUNE hd_cola_npi_negdet ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"hd_cola_npi_negdet\", pretrain_tasks = \"hd_cola_npi_negdet\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_negdet,hd_cola_npi_negdet,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_negdet
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_negdet_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_negdet, target_tasks = \"cola_npi_negdet,hd_cola_npi_negdet,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_negdet/model_state_hd_cola_npi_negdet_best.th\", use_classifier=\"hd_cola_npi_negdet\"" 

#### FINETUNE hd_cola_npi_cond ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"hd_cola_npi_cond\", pretrain_tasks = \"hd_cola_npi_cond\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_cond,hd_cola_npi_cond,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_cond
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_cond_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_cond
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_cond, target_tasks = \"cola_npi_cond,hd_cola_npi_cond,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_cond/model_state_hd_cola_npi_cond_best.th\", use_classifier=\"hd_cola_npi_cond\"" 

#### FINETUNE hd_cola_npi_adv ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"hd_cola_npi_adv\", pretrain_tasks = \"hd_cola_npi_adv\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_adv,hd_cola_npi_adv,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_adv
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_hd_cola_npi_adv_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_adv
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_hd_cola_npi_adv, target_tasks = \"cola_npi_adv,hd_cola_npi_adv,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_hd_cola_npi_adv/model_state_hd_cola_npi_adv_best.th\", use_classifier=\"hd_cola_npi_adv\"" 

#### FINETUNE all_cola_npi ####
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_model, target_tasks = \"all_cola_npi\", pretrain_tasks = \"all_cola_npi\", do_pretrain = 0, transfer_paradigm = finetune, do_full_eval = 0" 

#### EVAL cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,all_cola_npi,wilcox_npi ####
mkdir $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_all_cola_npi
mv $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_model/model_state_all_cola_npi_best.th $JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_all_cola_npi
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bertmnli, run_name = run_bertmnli_all_cola_npi, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv,all_cola_npi,wilcox_npi\", pretrain_tasks = \"none\", do_pretrain = 0, transfer_paradigm = finetune, do_target_task_training = 0, load_target_train_checkpoint = \"$JIANT_PROJECT_PREFIX/npi_bertmnli/run_bertmnli_all_cola_npi/model_state_all_cola_npi_best.th\", use_classifier=\"all_cola_npi\"" 

