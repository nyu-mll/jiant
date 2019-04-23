#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --job-name=myrun
#SBATCH --output=slurm_%j.out

python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_model, target_tasks = \"mnli\", pretrain_tasks = \"mnli\", do_pretrain = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola, target_tasks = \"cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola_npi_adv
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_adv, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola_npi_cond
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_cond, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_cond\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_negdet, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_negdet\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_negsent, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_negsent\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola_npi_only
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_only, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_only\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_qnt, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_qnt\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola_npi_ques
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_ques, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_ques\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_quessmp, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_quessmp\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_hd_cola_npi_sup
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_sup, target_tasks = \"cola_npi_sup\", pretrain_tasks = \"cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_cola_npi_sup
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_cola_npi_sup, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_sup\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_hd_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_quessmp, target_tasks = \"cola_npi_quessmp\", pretrain_tasks = \"cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_hd_cola_npi_ques
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_ques, target_tasks = \"cola_npi_ques\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_hd_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_qnt, target_tasks = \"cola_npi_qnt\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_hd_cola_npi_only
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_only, target_tasks = \"cola_npi_only\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_hd_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_negsent, target_tasks = \"cola_npi_negsent\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_hd_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_negdet, target_tasks = \"cola_npi_negdet\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_hd_cola_npi_cond
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_cond, target_tasks = \"cola_npi_cond\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_hd_cola_npi_adv
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_hd_cola_npi_adv, target_tasks = \"cola_npi_adv\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_model $JIANT_PROJECT_PREFIX/npi_bert/run_bertmnli_allnpi
python main.py --config_file config/spring19_seminar/bert.conf \
    --overrides "exp_name = npi_bert, run_name = run_bertmnli_allnpi, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_model, target_tasks = \"mnli\", pretrain_tasks = \"mnli\", do_pretrain = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola, target_tasks = \"cola,cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola_npi_adv
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_adv, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola_npi_cond
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_cond, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_cond\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_negdet, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_negdet\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_negsent, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_negsent\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola_npi_only
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_only, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_only\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_qnt, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_qnt\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola_npi_ques
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_ques, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_ques\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_quessmp, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_quessmp\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_hd_cola_npi_sup
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_sup, target_tasks = \"cola_npi_sup\", pretrain_tasks = \"cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_cola_npi_sup
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_cola_npi_sup, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_sup\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_hd_cola_npi_quessmp
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_quessmp, target_tasks = \"cola_npi_quessmp\", pretrain_tasks = \"cola_npi_sup,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_hd_cola_npi_ques
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_ques, target_tasks = \"cola_npi_ques\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_hd_cola_npi_qnt
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_qnt, target_tasks = \"cola_npi_qnt\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_hd_cola_npi_only
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_only, target_tasks = \"cola_npi_only\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_hd_cola_npi_negsent
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_negsent, target_tasks = \"cola_npi_negsent\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_hd_cola_npi_negdet
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_negdet, target_tasks = \"cola_npi_negdet\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_hd_cola_npi_cond
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_cond, target_tasks = \"cola_npi_cond\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_hd_cola_npi_adv
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_hd_cola_npi_adv, target_tasks = \"cola_npi_adv\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond\", do_pretrain = 1, load_model = 1" 

cp -r $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_model $JIANT_PROJECT_PREFIX/npi_bow_glove/run_bow_glovemnli_allnpi
python main.py --config_file config/spring19_seminar/bow_glove.conf \
    --overrides "exp_name = npi_bow_glove, run_name = run_bow_glovemnli_allnpi, target_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", pretrain_tasks = \"cola_npi_sup,cola_npi_quessmp,cola_npi_ques,cola_npi_qnt,cola_npi_only,cola_npi_negsent,cola_npi_negdet,cola_npi_cond,cola_npi_adv\", do_pretrain = 1, load_model = 1" 

