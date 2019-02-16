#!/bin/bash

source bert_user_config.sh

gpuid=${2:-0}

### FINAL EXPERIMENTS ###

## Debugging ##
function debug() {
    python -m ipdb main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, target_tasks = \"sst,mrpc\", do_pretrain = 0, do_target_task_training = 1, do_full_eval = 1, max_epochs_per_task = 3, transfer_paradigm = finetune, mnli-alt_pair_attn = 0, run_name = debug-bert, bert_embeddings_mode = \"top\", sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

## GLUE pretraining ##
function fullbert_mnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function wordbert_mnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-nobert, bert_embeddings_mode = only, cuda = ${gpuid}"
} 

function fullbert_qqp() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = qqp-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function wordbert_qqp() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = qqp-nobert, bert_embeddings_mode = only, cuda = ${gpuid}"
}

function fullbert_qnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qnli-alt, qnli-alt_pair_attn = 0, run_name = qnli-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wordbert_qnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qnli-alt, qnli-alt_pair_attn = 0, run_name = qnli-nobert, bert_embeddings_mode = only, cuda = ${gpuid}"
}

#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-bert, bert_embeddings_mode = none, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = qnli-alt, qnli-alt_pair_attn = 0, run_name = qnli-bert, bert_embeddings_mode = none, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = qqp-bert, bert_embeddings_mode = none, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = cola, run_name = cola-bert, bert_embeddings_mode = none, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = sst, run_name = sst-bert, bert_embeddings_mode = none, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = rte, run_name = rte-bert, bert_embeddings_mode = none, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = wnli, run_name = wnli-bert, bert_embeddings_mode = none, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = mrpc, run_name = mrpc-bert, bert_embeddings_mode = none, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 0, train_for_eval = 1, pretrain_tasks = cola, run_name = cola-bert, bert_embeddings_mode = none, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = sts-b-alt, sts-b-alt_pair_attn = 0, run_name = sts-b-bert, bert_embeddings_mode = none, cuda = 3"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 0, train_for_eval = 1, pretrain_tasks = rte, run_name = rte-bert, bert_embeddings_mode = none, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 0, train_for_eval = 1, pretrain_tasks = wnli, run_name = wnli-bert, bert_embeddings_mode = none, cuda = 2"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 0, train_for_eval = 1, pretrain_tasks = mrpc, run_name = mrpc-bert, bert_embeddings_mode = none, cuda = 5"

#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-nobert, bert_embeddings_mode = only, cuda = 6"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = qnli-alt, qnli-alt_pair_attn = 0, run_name = qnli-nobert, bert_embeddings_mode = only, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = qqp-nobert, bert_embeddings_mode = only, cuda = 6"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = cola, run_name = cola-nobert, bert_embeddings_mode = only, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = sst, run_name = sst-nobert, bert_embeddings_mode = only, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = rte, run_name = rte-nobert, bert_embeddings_mode = only, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = wnli, run_name = wnli-nobert, bert_embeddings_mode = only, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 1, train_for_eval = 0, pretrain_tasks = mrpc, run_name = mrpc-nobert, bert_embeddings_mode = only, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 0, train_for_eval = 1, pretrain_tasks = cola, run_name = cola-nobert, bert_embeddings_mode = only, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = sts-b-alt, sts-b-alt_pair_attn = 0, run_name = sts-b-nobert, bert_embeddings_mode = only, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 0, train_for_eval = 1, pretrain_tasks = rte, run_name = rte-nobert, bert_embeddings_mode = only, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 0, train_for_eval = 1, pretrain_tasks = wnli, run_name = wnli-nobert, bert_embeddings_mode = only, cuda = 2"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, do_train = 0, train_for_eval = 1, pretrain_tasks = mrpc, run_name = mrpc-nobert, bert_embeddings_mode = only, cuda = 5"


# Reruns for variance
partition1="cola,mnli,mrpc,rte,wnli"
partition2="qqp,sst"
partition3="qnli,sts-b"
partition23="qqp,qnli,sst,sts-b"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-nobert-p1-s2, bert_embeddings_mode = only, random_seed = 2222, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-nobert-p1-s3, bert_embeddings_mode = only, random_seed = 3333, cuda = 1"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-nobert-p1-s4, bert_embeddings_mode = only, random_seed = 4444, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-nobert-p1-s5, bert_embeddings_mode = only, random_seed = 5555, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-bert-p1-s2, bert_embeddings_mode = none, random_seed = 2222"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-bert-p1-s3, bert_embeddings_mode = none, random_seed = 3333"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-bert-p1-s4, bert_embeddings_mode = none, random_seed = 4444"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-bert-p1-s5, bert_embeddings_mode = none, random_seed = 5555"
partition0="mnli-alt"
ckpt2="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/friends-bert/mnli-nobert-p1-s2/model_state_main_epoch_60.best_macro.th"
ckpt3="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/friends-bert/mnli-nobert-p1-s3/model_state_main_epoch_73.best_macro.th"
ckpt4="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/friends-bert/mnli-nobert-p1-s4/model_state_main_epoch_64.best_macro.th"
ckpt5="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/friends-bert/mnli-nobert-p1-s5/model_state_main_epoch_50.best_macro.th"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"${partition0}\", eval_tasks = \"${partition2}\", do_train = 0, train_for_eval = 1, do_eval = 1, mnli-alt_pair_attn = 0, run_name = mnli-nobert-p2-s4, bert_embeddings_mode = only, load_eval_checkpoint = ${ckpt4}, random_seed = 4444, cuda = 3"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"${partition0}\", eval_tasks = \"${partition2}\", do_train = 0, train_for_eval = 1, do_eval = 1, mnli-alt_pair_attn = 0, run_name = mnli-nobert-p2-s5, bert_embeddings_mode = only, load_eval_checkpoint = ${ckpt5}, random_seed = 5555, cuda = 1"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"${partition0}\", eval_tasks = \"${partition3}\", do_train = 0, train_for_eval = 1, do_eval = 1, mnli-alt_pair_attn = 0, run_name = mnli-nobert-p3-s2, bert_embeddings_mode = only, load_eval_checkpoint = ${ckpt2}, random_seed = 2222, cuda = 3"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"${partition0}\", eval_tasks = \"${partition3}\", do_train = 0, train_for_eval = 1, do_eval = 1, mnli-alt_pair_attn = 0, run_name = mnli-nobert-p3-s3, bert_embeddings_mode = only, load_eval_checkpoint = ${ckpt3}, random_seed = 3333, cuda = 3"

## Random encoder ##
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-nobert-s3, bert_embeddings_mode = only, allow_missing_task_map = 1, random_seed = 91011, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-nobert-s4, bert_embeddings_mode = only, allow_missing_task_map = 1, random_seed = 121314, cuda = 7"

#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert, bert_embeddings_mode = none, allow_missing_task_map = 1, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition1}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p1-s2, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 2222, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition1}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p1-s3, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 3333, cuda = 0"

#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition2}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p2-s2, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 2222, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition2}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p2-s3, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 3333, cuda = 2"

#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition3}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p3-s2, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 2222, cuda = 2"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition3}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p3-s3, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 3333, cuda = 2"

#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition1}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p1-s4, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 4444, cuda = 2"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition1}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p1-s5, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 5555, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition2}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p2-s4, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 4444, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition2}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p2-s5, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 5555, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition3}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p3-s4, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 4444, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = none, eval_tasks = \"${partition3}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-bert-p3-s5, bert_embeddings_mode = none, allow_missing_task_map = 1, random_seed = 5555, cuda = 7"


## Reddit ##
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_pair_classif_3.4G, run_name = reddit-class-bert, bert_embeddings_mode = none, pair_attn = 0, cuda = 1"


## DisSent ##
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = dissentwikifullbig, run_name = dissent-bert, bert_embeddings_mode = none, pair_attn = 0, cuda = 3"


## Grounded ##
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = grounded, run_name = grounded-bert, bert_embeddings_mode = none, cuda = 4"


## Translation: no attn ##
# ELMo
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-noattn-bert, bert_embeddings_mode = none, lr = 0.001, max_grad_norm = 1.0, wmt17_en_ru_s2s_attention = none, max_seq_len = 64, batch_size = 32, cuda = 5"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-noattn-bert, bert_embeddings_mode = none, lr = 0.001, max_grad_norm = 1.0, wmt14_en_de_s2s_attention = none, max_seq_len = 64, batch_size = 32, cuda = 2"
# no ELMo
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-noattn-nobert, bert_embeddings_mode = only, lr = 0.001, max_grad_norm = 1.0, wmt17_en_ru_s2s_attention = none, max_seq_len = 64, batch_size = 32, cuda = 4"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-noattn-nobert, bert_embeddings_mode = only, lr = 0.001, max_grad_norm = 1.0, wmt14_en_de_s2s_attention = none, max_seq_len = 64, batch_size = 32, cuda = 7"

## S2S stuff
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = wiki103_s2s, run_name = wiki103-s2s-attn-nobert, bert_embeddings_mode = only, lr = 0.001, wiki103_s2s_s2s_attention = bilinear, max_seq_len = 64, cuda = N"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = wiki103_s2s, run_name = wiki103-s2s-attn-bert, bert_embeddings_mode = none, lr = 0.001, wiki103_s2s_s2s_attention = bilinear, max_seq_len = 64, cuda = N"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-attn-nobert, bert_embeddings_mode = only, lr = 0.001, max_grad_norm = 1.0, reddit_s2s_3.4G_s2s_attention = bilinear, max_seq_len = 64, cuda = N"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-attn-bert, bert_embeddings_mode = none, lr = 0.001, max_grad_norm = 1.0, reddit_s2s_3.4G_s2s_attention = bilinear, max_seq_len = 64, cuda = 3"

## LM with 20k vocab ##
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = bwb, run_name = bwb-lm-nobert, bert_embeddings_mode = only, lr = 0.001, cuda = 1"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = wiki103, run_name = wiki103-lm-nobert, bert_embeddings_mode = only, lr = 0.001, cuda = 1"


## GLUE MTL ##
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, run_name = mtl-glue-bert, bert_embeddings_mode = none, do_train = 1, train_for_eval = 0, cuda = 6"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, run_name = mtl-glue-bert, bert_embeddings_mode = none, do_train = 0, train_for_eval = 1, cuda = 2"


# non GLUE MTL
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 9000, run_name = mtl-nonglue-all-nobert, bert_embeddings_mode = only, dec_val_scale = 250, cuda = 1"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 7000, run_name = mtl-nonglue-nolm-nobert, bert_embeddings_mode = only, dec_val_scale = 250, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 7000, run_name = mtl-nonglue-nolm-bert, elmo_chars_only = 0, seq_embs_for_skip = 1, dec_val_scale = 250, batch_size = 32, cuda = 3"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 9000, run_name = mtl-nonglue-all-nobert-highlr, bert_embeddings_mode = only, dec_val_scale = 250, cuda = 5, lr = .001"

# all MTL
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 18000, run_name = mtl-alltasks-all-nobert, bert_embeddings_mode = only, dec_val_scale = 250, do_train = 0, train_for_eval = 1, cuda = 1"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 16000, run_name = mtl-alltasks-nolm-nobert, bert_embeddings_mode = only, dec_val_scale = 250, do_train = 0, train_for_eval = 1, cuda = 6"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 16000, run_name = mtl-alltasks-nolm-bert, bert_embeddings_mode = none, dec_val_scale = 250, do_train = 0, train_for_eval = 1, cuda = 1"

partition1="qqp,rte,sst,sts-b,wnli"
partition2="cola,mnli,mrpc,qnli"
eval_ckpt="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/friends-bert/mtl-alltasks-nolm-bert/model_state_main_epoch_34.best_macro.th"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded,${partition2}\", eval_tasks = \"${partition1}\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 16000, run_name = mtl-alltasks-nolm-bert-p2, bert_embeddings_mode = none, dec_val_scale = 250, do_train = 0, train_for_eval = 1, load_eval_checkpoint = ${eval_ckpt}, cuda = 6"



### Learning curves ###

## Reddit
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = 0.05719, run_name = reddit_s2s-1024k-bert, bert_embeddings_mode = none, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00357, run_name = reddit_s2s-256k-bert, bert_embeddings_mode = none, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00357, run_name = reddit_s2s-64k-bert, bert_embeddings_mode = none, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64, cuda = 6"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00089, run_name = reddit_s2s-16k-bert, bert_embeddings_mode = none, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64, cuda = 6"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00089, run_name = reddit_pair_classif-16k-bert, bert_embeddings_mode = none, pair_attn = 0, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00357, run_name = reddit_pair_classif-64k-bert, bert_embeddings_mode = none, pair_attn = 0, cuda = 7"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.01430, run_name = reddit_pair_classif-256k-bert, bert_embeddings_mode = none, pair_attn = 0, cuda = 2"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.05719, run_name = reddit_pair_classif-1024k-bert, bert_embeddings_mode = none, pair_attn = 0, cuda = 2"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00089, run_name = reddit_pair_classif-16k-bert, bert_embeddings_mode = none, pair_attn = 0, do_train = 0, cuda = 5"

## Target task learning curve
## MTL
run_dir="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/friends-bert/mtl-nonglue-all-nobert-copy/"
model_state_file="model_state_main_epoch_176.best_macro.th"

# MNLI
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.00255, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-1k-mtl, bert_embeddings_mode = only, dec_val_scale = 250"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.01019, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-4k-mtl, bert_embeddings_mode = only, dec_val_scale = 250"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.04074, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-16k-mtl, bert_embeddings_mode = only, dec_val_scale = 250"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.16297, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-64k-mtl, bert_embeddings_mode = only, dec_val_scale = 250, cuda = 0"

# QQP
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.00275, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-1k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.01099, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-4k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.04397, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-16k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.17589, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-64k-mtl, bert_embeddings_mode = only, dec_val_scale = 250, cuda = 2" 

# SST
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.01485, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-1k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.05939, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-4k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.23757, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-16k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.47514, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-32k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 

# STS-B
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.17391, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-1k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.69565, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-4k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 

# CoLA
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.11695, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-1k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.46778, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-4k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 

# MRPC
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.27255, do_train = 0, eval_tasks = mrpc, do_eval = 1, run_name = lc-target-mrpc-1k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 

# QNLI
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.00922, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-1k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.03689, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-4k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.14755, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-16k-mtl, bert_embeddings_mode = only, dec_val_scale = 250" 
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.59021, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-64k-mtl, bert_embeddings_mode = only, dec_val_scale = 250, cuda = 7" 

if [ $1 == "debug" ]; then
    debug
elif [ $1 == "fullbert-mnli" ]; then
    fullbert_mnli
elif [ $1 == "wordbert-mnli" ]; then
    wordbert_mnli
fi

### Eval a model: rerun test ###
eval_cmd="do_train = 0, train_for_eval = 0, do_eval = 1, batch_size = 128, write_preds = test, write_strict_glue_format = 1"

#run_dir="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/final/mtl-glue-bert-v2-preds/"
# We could load run specific params from run_dir, but I'm just copying the original cmd with overrides.
#python main.py --config ${run_dir}/params.conf --overrides "exp_name = friends-bert, pretrain_tasks = bwb, run_name = bwb-lm-nobert, bert_embeddings_mode = only, lr = 0.001, ${eval_cmd}"

#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = bwb, run_name = bwb-lm-nobert, bert_embeddings_mode = only, lr = 0.001, ${eval_cmd}"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 7000, run_name = mtl-nonglue-nolm-bert, elmo_chars_only = 0, seq_embs_for_skip = 1, dec_val_scale = 250, ${eval_cmd}"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 18000, run_name = mtl-alltasks-all-nobert, bert_embeddings_mode = only, dec_val_scale = 250, ${eval_cmd}"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 16000, run_name = mtl-alltasks-nolm-bert, bert_embeddings_mode = none, dec_val_scale = 250, ${eval_cmd}"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 16000, run_name = mtl-alltasks-nolm-bert-p2, bert_embeddings_mode = none, dec_val_scale = 250, ${eval_cmd}"
