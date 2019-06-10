#!/bin/bash

# This file defines experiments to recreate the learning curve figures (Figures 2 and 3).
# To run, either use `pretraining_bert_exps.sh` (e.g. `> pretraining_bert_exps.sh mnli_lc`)
# or source this file and run a command from it (e.g. `source pretraining_bert_lc_exps.sh && mnli_lc`).

# N examples per task:
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

#########################################
##  Intermediate task training curves  ##
#########################################
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
    qnli_100k
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
    sst_64k
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
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki103_s2s, training_data_fraction = .00402, run_name = wiki-s2s-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki103_s2s, training_data_fraction = .00804, run_name = wiki-s2s-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki103_s2s, training_data_fraction = .01609, run_name = wiki-s2s-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki103_s2s, training_data_fraction = .02514, run_name = wiki-s2s-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wiki103_s2s, training_data_fraction = .06435, run_name = wiki-s2s-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function wiki_s2s_lc() {
    wiki_s2s_16k
    wiki_s2s_32k
    wiki_s2s_64k
    wiki_s2s_100k
    wiki_s2s_256k
}

function reddit_s2s_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .00089, run_name = reddit_s2s-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .00179, run_name = reddit_s2s-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .00357, run_name = reddit_s2s-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .00559, run_name = reddit_s2s-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_s2s_3.4G, training_data_fraction = .01430, run_name = reddit_s2s-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function reddit_s2s_lc() {
    reddit_s2s_16k
    reddit_s2s_32k
    reddit_s2s_64k
    reddit_s2s_100k
    reddit_s2s_256k
}

function mt_de_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .00466, run_name = mt_de-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .00931, run_name = mt_de-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .01863, run_name = mt_de-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .02910, run_name = mt_de-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt14_en_de, training_data_fraction = .07450, run_name = mt_de-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_de_lc() {
    mt_de_16k
    mt_de_32k
    mt_de_64k
    mt_de_100k
    mt_de_256k
}

function mt_ru_16k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt17_en_ru, training_data_fraction = .00503, run_name = mt-ru-16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_ru_32k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt17_en_ru, training_data_fraction = .01006, run_name = mt-ru-32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_ru_64k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt17_en_ru, training_data_fraction = .02013, run_name = mt-ru-64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_ru_100k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt17_en_ru, training_data_fraction = .03145, run_name = mt-ru-100k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_ru_256k() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = wmt17_en_ru, training_data_fraction = .08050, run_name = mt-ru-256k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}
function mt_ru_lc() {
    mt_ru_16k
    mt_ru_32k
    mt_ru_64k
    mt_ru_100k
    mt_ru_256k
}

#####################################
## Target task learning curve runs ##
#####################################
function target_lc_mnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-mnli1k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = mnli, eval_data_fraction = 0.00255, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-mnli4k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = mnli, eval_data_fraction = 0.01019, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-mnli16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = mnli, eval_data_fraction = 0.04074, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-mnli64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = mnli, eval_data_fraction = 0.16297, cuda = ${gpuid}"
}

function target_lc_qqp() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-qqp1k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = qqp, eval_data_fraction = 0.00275, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-qqp4k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = qqp, eval_data_fraction = 0.01099, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-qqp16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = qqp, eval_data_fraction = 0.04397, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-qqp64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = qqp, eval_data_fraction = 0.17589, cuda = ${gpuid}"
}

function target_lc_qnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-qnli1k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = qnli, eval_data_fraction = 0.00922, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-qnli4k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = qnli, eval_data_fraction = 0.03689, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-qnli16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = qnli, eval_data_fraction = 0.14755, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-qnli64k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = qnli, eval_data_fraction = 0.59021, cuda = ${gpuid}"
}

function target_lc_sst() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-sst1k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = sst, eval_data_fraction = 0.01485, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-sst4k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = sst, eval_data_fraction = 0.05939, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-sst16k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = sst, eval_data_fraction = 0.23757, cuda = ${gpuid}"
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, run_name = trg-lc-sst32k, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, do_pretrain = 0, target_tasks = sst, eval_data_fraction = 0.47514, cuda = ${gpuid}"
}


