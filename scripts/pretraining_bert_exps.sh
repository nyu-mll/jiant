#!/bin/bash

source user_config.sh
source scripts/pretraining_bert_lc_exps.sh

gpuid=${2:-0}

## Just BERT, no additional training ##
function fullbert() {
    python main.py --config config/final-bert.conf --overrides "do_pretrain = 0, allow_untrained_encoder_parameters = 1, run_name = untrained-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

######################
## GLUE pretraining ##
######################

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


######################
##      DisSent     ##
######################
function fullbert_dissent() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = dissentwikifullbig, run_name = dissent-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

###########################
## Reddit classification ##
###########################
function fullbert_reddit_clf() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = reddit_pair_classif_3.4G, run_name = preproc-reddit-clf, do_pretrain = 0, do_target_task_training = 0, do_full_eval = 0, bert_embeddings_mode = none, pair_attn = 0, cuda = ${gpuid}"
}

######################
##     S2S stuff    ##
######################
# NOTE: These runs are in a different experiment directory ("exp_name = XXX")
# because we need to build a target vocabulary for each task.
# To use these tasks in MTL experiments, we copied the 
# vocabulary and preprocessing objects from their experiment
# directories into a shared directory (defined in `config/final-bert.conf`).

# SkipThought #
function fullbert_skipthought() {
    python main.py --config config/final-bert.conf --overrides "exp_name = s2s-wiki, pretrain_tasks = wiki103_s2s, run_name = wiki103_s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

# Reddit seq2seq #
function fullbert_reddit_s2s() {
    python main.py --config config/final-bert.conf --overrides "exp_name = s2s-wiki, pretrain_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

# Translation: no attn #
# These `preproc_*` experiments just run the preprocessing for the MT experiments.
# It is sometimes useful to start these in advance because they can take a while.
# You can also just run the fullbert_mt_{de,ru} experiments and they will also
# first run the preprocessing if the files are not there already.
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

######################
##     GLUE MTL     ##
######################
function fullbert_glue() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"mnli-alt,mrpc-alt,qnli-alt,sst-alt,sts-b-alt,rte-alt,wnli-alt,qqp-alt,cola-alt\", run_name = mtl-glue-topbert-v2, mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

######################
##   non GLUE MTL   ##
######################
function fullbert_nonglue() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G\", val_interval = 5000, run_name = mtl-nonglue-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

######################
##      all MTL     ##
######################
function fullbert_all() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"mnli-alt,mrpc-alt,qnli-alt,sst-alt,sts-b-alt,rte-alt,wnli-alt,qqp-alt,cola-alt,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G\", val_interval = 14000, run_name = mtl-all-topbert-v2, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, mnli-alt_pair_attn = 0, mrpc-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b_pair_attn = 0, qqp_pair_attn = 0, wnli-alt_pair_attn = 0, cuda = ${gpuid}"
}

#########################
## Learning curve runs ##
#########################
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

##############################
## Example using BERT large ##
##############################
function fullbert_large() {
    python main.py --config config/final-bert.conf --overrides "tokenizer = \"bert-large-cased\", bert_model_name = \"bert-large-cased\", batch_size = 8, exp_name = \"bert-large-cased\", do_pretrain = 0, allow_untrained_encoder_parameters = 1, run_name = untrained-topbert, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, cuda = ${gpuid}"
}

########################################################
## Example using only the *word* embeddings from BERT ##
########################################################
function wordbert_mnli() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-nobert, bert_embeddings_mode = only, cuda = ${gpuid}"
} 

#############################################
## Example only evaluating a trained model ##
#############################################
function fullbert_all_eval() {
    python main.py --config config/final-bert.conf --overrides "pretrain_tasks = \"mnli-alt,mrpc-alt,qnli-alt,sst-alt,sts-b-alt,rte-alt,wnli-alt,qqp-alt,cola-alt,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G\", val_interval = 14000, run_name = mtl-all-topbert-v4, bert_embeddings_mode = top, sent_enc = \"null\", sep_embs_for_skip = 1, mnli-alt_pair_attn = 0, mrpc-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b_pair_attn = 0, qqp_pair_attn = 0, wnli-alt_pair_attn = 0, do_pretrain = 0, cuda = ${gpuid}"
}



if [ $1 == "wordbert-mnli" ]; then
    wordbert_mnli
elif [ $1 == "fullbert-large" ]; then
    fullbert_large
elif [ $1 == "fullbert-all-eval" ]; then
    fullbert_all_eval
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
elif [ $1 == "wiki-s2s-lc" ]; then
    wiki_s2s_lc
elif [ $1 == "reddit-s2s-lc" ]; then
    reddit_s2s_lc
elif [ $1 == "mt-de-lc" ]; then
    mt_de_lc
elif [ $1 == "mt-ru-lc" ]; then
    mt_ru_lc
elif [ $1 == "target-lc-mnli" ]; then
    target_lc_mnli
elif [ $1 == "target-lc-qqp" ]; then
    target_lc_qqp
elif [ $1 == "target-lc-qnli" ]; then
    target_lc_qnli
elif [ $1 == "target-lc-sst" ]; then
    target_lc_sst
fi
