#!/bin/bash

source bert_user_config.sh

gpuid=${2:-0}

### FINAL EXPERIMENTS ###

## Debugging ##
function debug() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"cola-alt,mrpc-alt\", target_tasks = \"sst,mrpc\", do_pretrain = 1, do_target_task_training = 1, do_full_eval = 1, max_epochs_per_task = 1, transfer_paradigm = finetune, mnli-alt_pair_attn = 0, run_name = debug-multitask-finetune, bert_embeddings_mode = \"top\", sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function debug_large() {
    python main.py --config config/final-bert.conf --overrides "tokenizer = \"bert-large-cased\", bert_model_name = \"bert-large-cased\", exp_name = \"bert-large-cased\", pretrain_tasks = \"mnli-alt\", do_pretrain = 1, max_epochs_per_task = 1, transfer_paradigm = finetune, mnli-alt_pair_attn = 0, run_name = debug-large, bert_embeddings_mode = \"top\", sent_enc = \"null\", sep_embs_for_skip = 1, batch_size = 16, cuda = ${gpuid}"
}

## GLUE pretraining ##
function fullbert_large() {
    python main.py --config config/final-bert.conf --overrides "tokenizer = \"bert-large-cased\", bert_model_name = \"bert-large-cased\", exp_name = \"bert-large-cased\", do_pretrain = 0, allow_untrained_encoder_parameters = 1, run_name = untrained-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert() {
    python main.py --config config/final-bert.conf --overrides "do_pretrain = 0, allow_untrained_encoder_parameters = 1, run_name = untrained-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

## Lexical BERT ##
function wordbert_mnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-nobert, bert_embeddings_mode = only, cuda = ${gpuid}"
} 

## GLUE pretraining ##
function fullbert_mnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = mnli-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert_qqp() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qqp-alt, run_name = qqp-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert_qnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qnli-alt, run_name = qnli-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert_stsb() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = sts-b-alt, run_name = sts-b-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function fullbert_cola() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = cola-alt, run_name = cola-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert_sst() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = sst-alt, run_name = sst-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert_mrpc() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mrpc-alt, run_name = mrpc-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert_rte() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = rte-alt, run_name = rte-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert_wnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wnli-alt, run_name = wnli-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}



## DisSent ##
function fullbert_dissent() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = dissentwikifullbig, run_name = dissent-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

## Reddit classification ##
function fullbert_reddit_clf() {
    python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_pair_classif_3.4G, run_name = reddit-class-bert, bert_embeddings_mode = none, pair_attn = 0, cuda = 1"
}

## S2S stuff

# SkipThought #
function fullbert_skipthought(){
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki103_s2s, run_name = wiki103_s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, wiki103_s2s_s2s_attention = none, cuda = ${gpuid}"
}

# Reddit seq2seq #
function fullbert_reddit_s2s() {
    #max_grad_norm = 1.0
    #max_seq_len = 64
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, reddit_s2s_3.4G_s2s_attention = none, cuda = ${gpuid}"
}

# Translation: no attn #
function fullbert_mt_ru() {
    python main.py --config config/final-bert.conf --overrides "exp_name = mt-en-ru, pretrain_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, wmt17_en_ru_s2s_attention = none, cuda = ${gpuid}"
}

function fullbert_mt_de() {
    python main.py --config config/final-bert.conf --overrides "exp_name = mt-en-de, pretrain_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, wmt14_en_de_s2s_attention = none, cuda = ${gpuid}"
}

## GLUE MTL ##
function fullbert_glue() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"mnli-alt,mrpc-alt,qnli-alt,sst-alt,sts-b-alt,rte-alt,wnli-alt,qqp-alt,cola-alt\", run_name = mtl-glue-topbert, mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}



# non GLUE MTL
function fullbert_nonglue() {
    echo "not implemented"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 9000, run_name = mtl-nonglue-all-nobert, bert_embeddings_mode = only, dec_val_scale = 250, cuda = 1"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 7000, run_name = mtl-nonglue-nolm-nobert, bert_embeddings_mode = only, dec_val_scale = 250, cuda = 0"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 7000, run_name = mtl-nonglue-nolm-bert, elmo_chars_only = 0, seq_embs_for_skip = 1, dec_val_scale = 250, batch_size = 32, cuda = 3"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 9000, run_name = mtl-nonglue-all-nobert-highlr, bert_embeddings_mode = only, dec_val_scale = 250, cuda = 5, lr = .001"
}

# all MTL
function fullbert_all() {
    echo "not implemented"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 18000, run_name = mtl-alltasks-all-nobert, bert_embeddings_mode = only, dec_val_scale = 250, do_train = 0, train_for_eval = 1, cuda = 1"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 16000, run_name = mtl-alltasks-nolm-nobert, bert_embeddings_mode = only, dec_val_scale = 250, do_train = 0, train_for_eval = 1, cuda = 6"
#python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 16000, run_name = mtl-alltasks-nolm-bert, bert_embeddings_mode = none, dec_val_scale = 250, do_train = 0, train_for_eval = 1, cuda = 1"
}


if [ $1 == "debug" ]; then
    debug
elif [ $1 == "debug-large" ]; then
    debug_large
elif [ $1 == "wordbert-mnli" ]; then
    wordbert_mnli
elif [ $1 == "fullbert" ]; then
    fullbert
elif [ $1 == "fullbert-mnli" ]; then
    fullbert_mnli
elif [ $1 == "fullbert-qqp" ]; then
    fullbert_qqp
elif [ $1 == "fullbert-qnli" ]; then
    fullbert_qnli
elif [ $1 == "fullbert-stsb" ]; then
    fullbert_stsb
elif [ $1 == "fullbert-mrpc" ]; then
    fullbert_mrpc
elif [ $1 == "fullbert-rte" ]; then
    fullbert_rte
elif [ $1 == "fullbert-wnli" ]; then
    fullbert_wnli
elif [ $1 == "fullbert-cola" ]; then
    fullbert_cola
elif [ $1 == "fullbert-sst" ]; then
    fullbert_sst
elif [ $1 == "fullbert-dissent" ]; then
    fullbert_dissent
elif [ $1 == "fullbert-skipthought" ]; then
    fullbert_skipthought
elif [ $1 == "fullbert-reddit-clf" ]; then
    fullbert_reddit_clf
elif [ $1 == "fullbert-reddit-s2s" ]; then
    fullbert_reddit_s2s
elif [ $1 == "fullbert-mt-ru" ]; then
    fullbert_mt_ru
elif [ $1 == "fullbert-mt-de" ]; then
    fullbert_mt_de
elif [ $1 == "fullbert-glue" ]; then
    fullbert_glue
elif [ $1 == "fullbert-nonglue" ]; then
    fullbert_nonglue
elif [ $1 == "fullbert-glue" ]; then
    fullbert_all
elif [ $1 == "fullbert-large" ]; then
    fullbert_large
fi
