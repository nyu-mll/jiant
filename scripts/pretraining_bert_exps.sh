#!/bin/bash

source bert_user_config.sh

gpuid=${2:-0}

### FINAL EXPERIMENTS ###

## Debugging ##
function debug() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"cola-alt,mrpc-alt\", target_tasks = \"sst,mrpc\", do_pretrain = 1, do_target_task_training = 1, do_full_eval = 1, max_epochs_per_task = 1, transfer_paradigm = finetune, mnli-alt_pair_attn = 0, run_name = debug-multitask-finetune, bert_embeddings_mode = \"top\", sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function debug_large() {
    python main.py --config config/final-bert.conf --overrides "tokenizer = \"bert-large-cased\", bert_model_name = \"bert-large-cased\", exp_name = \"bert-large-cased\", pretrain_tasks = \"mnli-alt\", do_pretrain = 1, max_epochs_per_task = 1, transfer_paradigm = finetune, mnli-alt_pair_attn = 0, run_name = debug-large, bert_embeddings_mode = \"top\", sent_enc = \"null\", sep_embs_for_skip = 1, batch_size = 8, cuda = ${gpuid}"
}

function debug_s2s() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki_s2s_debug, target_tasks = \"none\", do_target_task_training = 0, do_full_eval = 0, run_name = debug-s2s, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, wiki103_s2s_s2s_attention = bilinear, max_grad_norm = 1.0, lr = .0000003, min_lr = .00000001, cuda = ${gpuid}"
}

## GLUE pretraining ##
function fullbert_large() {
    python main.py --config config/final-bert.conf --overrides "tokenizer = \"bert-large-cased\", bert_model_name = \"bert-large-cased\", batch_size = 8, exp_name = \"bert-large-cased\", do_pretrain = 0, allow_untrained_encoder_parameters = 1, run_name = untrained-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
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
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_pair_classif_3.4G, run_name = preproc-reddit-clf, do_pretrain = 0, do_target_task_training = 0, do_full_eval = 0, bert_embeddings_mode = none, pair_attn = 0, cuda = ${gpuid}"
    #python main.py --config config/final-bert.conf --overrides "exp_name = friends-bert, pretrain_tasks = reddit_pair_classif_3.4G, run_name = reddit-class-bert, bert_embeddings_mode = none, pair_attn = 0, cuda = ${gpuid}"
}

## S2S stuff
#max_grad_norm = 1.0
#max_seq_len = 64

# SkipThought #
function fullbert_skipthought() {
    python main.py --config config/final-bert.conf --overrides "exp_name = s2s-wiki, pretrain_tasks = wiki103_s2s, run_name = wiki103_s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

# Reddit seq2seq #
function fullbert_reddit_s2s() {
    python main.py --config config/final-bert.conf --overrides "exp_name = s2s-wiki, pretrain_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

# Translation: no attn #
function preproc_mt_ru(){ 
    python main.py --config config/final-bert.conf --overrides "exp_name = mt-ru, pretrain_tasks = \"wmt17_en_ru\", run_name = preproc-ru, do_pretrain = 0, do_target_task_training = 0, do_full_eval = 0, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function preproc_mt_de(){ 
    python main.py --config config/final-bert.conf --overrides "exp_name = mt-de, pretrain_tasks = \"wmt14_en_de\", run_name = preproc-de, do_pretrain = 0, do_target_task_training = 0, do_full_eval = 0, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert_mt_ru() {
    python main.py --config config/final-bert.conf --overrides "exp_name = mt-ru, pretrain_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function fullbert_mt_de() {
    python main.py --config config/final-bert.conf --overrides "exp_name = mt-de, pretrain_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

## GLUE MTL ##
function fullbert_glue() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"mnli-alt,mrpc-alt,qnli-alt,sst-alt,sts-b-alt,rte-alt,wnli-alt,qqp-alt,cola-alt\", run_name = mtl-glue-topbert, mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

## non GLUE MTL ##
function fullbert_nonglue() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G\", val_interval = 5000, run_name = mtl-nonglue-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

## all MTL ##
function fullbert_all() {
    #reddit_s2s_3.4G_s2s_attn = bilinear, wiki103_s2s_s2s_attn = bilinear
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"mnli-alt,mrpc-alt,qnli-alt,sst-alt,sts-b-alt,rte-alt,wnli-alt,qqp-alt,cola-alt,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G\", val_interval = 14000, run_name = mtl-all-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, mnli-alt_pair_attn = 0, mrpc-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b_pair_attn = 0, qqp_pair_attn = 0, wnli-alt_pair_attn = 0, cuda = ${gpuid}"
}


### Intermediate task training curves ###
# CoLA: 8551 -- none
# SST: 67349
# MRPC: 3668 -- none
# STS: 5748 -- none
# QQP: 363849
# MNLI: 392702
# QNLIv1: 108436
# RTE: 2490 -- none
# WNLI: 635 -- none
# MT EN-DE: 3436043
# MT EN-RU: 3180000
# DisSent: 311828
# Reddit s2s: 17903854
# Wiki s2s: 3978309

function mnli_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, training_data_fraction = .04074, run_name = mnli-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mnli_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, training_data_fraction = .08149, run_name = mnli-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mnli_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, training_data_fraction = .16297, run_name = mnli-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mnli_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, training_data_fraction = .25465, run_name = mnli-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mnli_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, training_data_fraction = .65189, run_name = mnli-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mnli_lc() {
    mnli_16k
    mnli_32k
    mnli_64k
    mnli_100k
    mnli_256k
}

function qqp_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qqp-alt, training_data_fraction = .04397, run_name = qqp-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function qqp_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qqp-alt, training_data_fraction = .08795, run_name = qqp-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function qqp_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qqp-alt, training_data_fraction = .17590, run_name = qqp-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function qqp_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qqp-alt, training_data_fraction = .27484, run_name = qqp-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function qqp_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qqp-alt, training_data_fraction = .70359, run_name = qqp-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function qqp_lc() {
    qqp_16k
    qqp_32k
    qqp_64k
    qqp_100k
    qqp_256k
}

function qnli_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qnli-alt, training_data_fraction = .14755, run_name = qnli-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function qnli_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qnli-alt, training_data_fraction = .29510, run_name = qnli-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function qnli_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qnli-alt, training_data_fraction = .59021, run_name = qnli-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function qnli_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = qnli-alt, training_data_fraction = .92220, run_name = qnli-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

function qnli_lc() {
    qnli_16k
    qnli_32k
    qnli_64k
    #qnli_100k
}

function sst_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = sst-alt, training_data_fraction = .23757, run_name = sst-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function sst_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = sst-alt, training_data_fraction = .47514, run_name = sst-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function sst_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = sst-alt, training_data_fraction = .95027, run_name = sst-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function sst_lc() {
    sst_16k
    sst_32k
    #sst_64k
}

function dissent_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = dissentwikifullbig, training_data_fraction = .05131, run_name = dissent-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function dissent_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = dissentwikifullbig, training_data_fraction = .10262, run_name = dissent-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function dissent_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = dissentwikifullbig, training_data_fraction = .20524, run_name = dissent-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function dissent_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = dissentwikifullbig, training_data_fraction = .32069, run_name = dissent-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function dissent_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = dissentwikifullbig, training_data_fraction = .82097, run_name = dissent-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function dissent_lc() {
    dissent_16k
    dissent_32k
    dissent_64k
    dissent_100k
    dissent_256k
}

function wiki_s2s_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki_s2s, training_data_fraction = .04074, run_name = wiki_s2s-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki_s2s, training_data_fraction = .08149, run_name = wiki_s2s-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki_s2s, training_data_fraction = .16297, run_name = wiki_s2s-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki_s2s, training_data_fraction = .25465, run_name = wiki_s2s-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki_s2s, training_data_fraction = .65189, run_name = wiki_s2s-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_lc() {
    wiki_s2s_16k
    wiki_s2s_32k
    wiki_s2s_64k
    wiki_s2s_100k
    wiki_s2s_256k
}

function reddit_s2s_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .04074, run_name = reddit_s2s-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .08149, run_name = reddit_s2s-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .16297, run_name = reddit_s2s-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .25465, run_name = reddit_s2s-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .65189, run_name = reddit_s2s-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_lc() {
    reddit_s2s_16k
    reddit_s2s_32k
    reddit_s2s_64k
    reddit_s2s_100k
    reddit_s2s_256k
}

function mt_de_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .04074, run_name = mt_de-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .08149, run_name = mt_de-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .16297, run_name = mt_de-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .25465, run_name = mt_de-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .65189, run_name = mt_de-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_lc() {
    mt_de_16k
    mt_de_32k
    mt_de_64k
    mt_de_100k
    mt_de_256k
}

function mt_de_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .04074, run_name = mt_de-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .08149, run_name = mt_de-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .16297, run_name = mt_de-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .25465, run_name = mt_de-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .65189, run_name = mt_de-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_lc() {
    mt_de_16k
    mt_de_32k
    mt_de_64k
    mt_de_100k
    mt_de_256k
}


if [ $1 == "debug" ]; then
    debug
elif [ $1 == "debug-large" ]; then
    debug_large
elif [ $1 == "debug-s2s" ]; then
    debug_s2s
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
elif [ $1 == "preproc-mt-ru" ]; then
    preproc_mt_ru
elif [ $1 == "preproc-mt-de" ]; then
    preproc_mt_de
elif [ $1 == "fullbert-mt-ru" ]; then
    fullbert_mt_ru
elif [ $1 == "fullbert-mt-de" ]; then
    fullbert_mt_de
elif [ $1 == "fullbert-glue" ]; then
    fullbert_glue
elif [ $1 == "fullbert-nonglue" ]; then
    fullbert_nonglue
elif [ $1 == "fullbert-all" ]; then
    fullbert_all
elif [ $1 == "mnli-lc" ]; then
    mnli_lc
elif [ $1 == "qqp-lc" ]; then
    qqp_lc
elif [ $1 == "qnli-lc" ]; then
    qnli_lc
elif [ $1 == "sst-lc" ]; then
    sst_lc
elif [ $1 == "dissent-lc" ]; then
    dissent_lc
elif [ $1 == "fullbert-large" ]; then
    fullbert_large
fi
