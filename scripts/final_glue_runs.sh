# These are the pretraining runs that will be included in our three main papers.

# Ground rules:
# - There will be a separate tab in the sheet for these results. All results in the paper will come from that tab,
#     and all results in that tab must be from the runs described here.
# - All runs must be from the master branch.
# - We will not change defaults.conf, to preserve compatibility with older runs. Shared defaults will go in final.conf.
# - You may not modify the overrides in these commands except through a reviewed pull request.
# - No pull requests will be allowed after 5p on Tuesday 7/24, except in cases of _mistakes_ in the commands below.
# - All results on the sheet must be reported through the script. You may not manually type anything in the main area of the sheet.
# - These commands are set up for NYU. You may, of course, modify everything outside the quotation marks to suit your machine setup.
#     Make sure you use final.conf if you do.
# - All runs will be started by 5p on Friday 7/27.

## GLUE tasks as pretraining ##

# Sam ran all 18 of these.
# Alex re-ran all 18 of these.
# TODO: Non-alt tasks will need to be run in two parts because of https://github.com/jsalt18-sentence-repl/jiant/issues/290

JIANT_OVERRIDES="train_tasks = cola, run_name = cola-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = cola, run_name = cola-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = sst, run_name = sst-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = sst, run_name = sst-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = rte, run_name = rte-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = rte, run_name = rte-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = wnli, run_name = wnli-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wnli, run_name = wnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = mrpc, run_name = mrpc-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = mrpc, run_name = mrpc-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = qnli-alt, qnli-alt_pair_attn = 0, run_name = qnli-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = qnli-alt, qnli-alt_pair_attn = 0, run_name = qnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = qqp-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = qqp-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = sts-b-alt, sts-b-alt_pair_attn = 0, run_name = sts-b-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = sts-b-alt, sts-b-alt_pair_attn = 0, run_name = sts-b-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch


## Random BiLSTM, no pretraining ##

# Sam ran these 2.
# Alex re-ran all 2 of these.
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Restarts for NLI probing.
JIANT_OVERRIDES="train_tasks = none, eval_tasks = mnli, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo-restart2, elmo_chars_only = 1, random_seed = 1111" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, eval_tasks = mnli, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo-restart3, elmo_chars_only = 1, random_seed = 2222" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, eval_tasks = mnli, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo-restart4, elmo_chars_only = 1, random_seed = 3333" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, eval_tasks = mnli, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo-restart5, elmo_chars_only = 1, random_seed = 4444" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch


## MT ##

# Seq2seq with attention.
# Katherin ran these.
# Alex is *not* rerunning these, based on early results indicating that MT without attention transferred better.
# NOTE: using a projection between decoder output and vocab softmax allows us to use the same batch size.
JIANT_OVERRIDES="train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, wmt17_en_ru_s2s_attention = bilinear, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, wmt17_en_ru_s2s_attention = bilinear, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, wmt14_en_de_s2s_attention = bilinear, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, wmt14_en_de_s2s_attention = bilinear, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Seq2seq no attention
# Katherin ran these.
# Alex is rerunning these.
JIANT_OVERRIDES="train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, wmt17_en_ru_s2s_attention = none, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, wmt17_en_ru_s2s_attention = none, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, wmt14_en_de_s2s_attention = none, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, wmt14_en_de_s2s_attention = none, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Reddit ##

# Seq2seq no attention
# Katherin ran these.
# Alex is *not* rerunning these, based on early results indicating that using attention trasferred better.
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, reddit_s2s_3.4G_s2s_attention = none, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, reddit_s2s_3.4G_s2s_attention = none, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Seq2seq with attention
# Katherin ran these.
# Alex is rerunning these.
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, reddit_s2s_3.4G_s2s_attention = bilinear, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, reddit_s2s_3.4G_s2s_attention = bilinear, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Classification

# Raghu ran these.
# Alex re-ran all 2 of these.
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, run_name = reddit-class-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, run_name = reddit-class-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## LM ##

# Standard LM training
# Note: ELMo can't combine with language modeling, so there are no ELMo runs.

# Alex ran these.
# Alex is now rerunning these.
JIANT_OVERRIDES="train_tasks = bwb, run_name = bwb-lm-noelmo, elmo_chars_only = 1, lr = 0.001" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103, run_name = wiki103-lm-noelmo, elmo_chars_only = 1, lr = 0.001" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Seq2seq (Skip-Thought), attn

# Katherin ran these.
# Alex is rerunning these.
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, wiki103_s2s_s2s_attention = bilinear, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, wiki103_s2s_s2s_attention = bilinear, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Seq2seq (Skip-Thought), no attention

# Katherin ran these.
# Alex is *not* rerunning these, based on early results indicating that using attention trasferred better.
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, wiki103_s2s_s2s_attention = none, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, wiki103_s2s_s2s_attention = none, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Classification (DiscSent-style. Which is not DisSent style. Aaagh.)

# Alex ran these.
JIANT_OVERRIDES="train_tasks = wiki103_classif, run_name = wiki103-cl-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103_classif, run_name = wiki103-cl-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## DisSent ##

# Alex ran these.
# Alex re-ran all 2 of these.
JIANT_OVERRIDES="train_tasks = dissentwikifullbig, run_name = dissent-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = dissentwikifullbig, run_name = dissent-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## MSCOCO ##

# Sam ran these.
# Alex re-ran all 2 of these.
JIANT_OVERRIDES="train_tasks = grounded, run_name = grounded-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = grounded, run_name = grounded-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## CCG (Note: For use in the NLI probing paper only) ##

# Alex re-ran all 2 of these.
JIANT_OVERRIDES="train_tasks = ccg, exp_name = final_ccg, run_name = ccg-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = ccg, exp_name = final_ccg, run_name = ccg-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## MTL ##

# GLUE MTL
# TODO: These will need to be run in two parts because of https://github.com/jsalt18-sentence-repl/jiant/issues/290

# Alex ran these.
# Alex is re-running all 2 of these.
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, run_name = mtl-glue-noelmo, elmo_chars_only = 1, do_train = 1, train_for_eval = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, run_name = mtl-glue-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, do_train = 1, train_for_eval = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, run_name = mtl-glue-noelmo, elmo_chars_only = 1, do_train = 0, train_for_eval = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, run_name = mtl-glue-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, do_train = 0, train_for_eval = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Non-GLUE MTL
# Alex: didn't find major differences with different weights of decreasing val metrics, so set dec_val_scale = 250
#   - wmt: wmt17_en_ru, wmt14_en_de
#   - reddit: reddit_s2s_3.4G, reddit_pair_classif_3.4G
#   - lm: wiki103, bwb
#   - skipthought: wiki103_s2s
#   - dissent: dissentwikifullbig
#   - grounded: grounded

# Alex tried to run these.
# Alex is now rerunning these.
# Monster run with everything we've got.
JIANT_OVERRIDES="train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 9000, run_name = mtl-nonglue-all-noelmo, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
# Do a run w/o LM so we can use full ELMo.
JIANT_OVERRIDES="train_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 7000, run_name = mtl-nonglue-nolm-noelmo, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 7000, run_name = mtl-nonglue-nolm-elmo, elmo_chars_only = 0, seq_embs_for_skip = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# All MTL: use all tasks. Take the val interval to be max of 10k
# Alex is now rerunning these.
# TODO: These will need to be run in two parts because of https://github.com/jsalt18-sentence-repl/jiant/issues/290
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 10000, run_name = mtl-alltasks-all-noelmo, elmo_chars_only = 1, dec_val_scale = 250, do_train = 1, train_for_eval = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 10000, run_name = mtl-alltasks-all-noelmo, elmo_chars_only = 1, dec_val_scale = 250, do_train = 0, train_for_eval = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# No LM, with and without ELMo
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 10000, run_name = mtl-alltasks-nolm-noelmo, elmo_chars_only = 1, dec_val_scale = 250, do_train = 1, train_for_eval = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 10000, run_name = mtl-alltasks-nolm-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, dec_val_scale = 250, do_train = 1, train_for_eval = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 10000, run_name = mtl-alltasks-nolm-noelmo, elmo_chars_only = 1, dec_val_scale = 250, do_train = 0, train_for_eval = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 10000, run_name = mtl-alltasks-nolm-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, dec_val_scale = 250, do_train = 0, train_for_eval = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Target task learning curves ##

## Source task learning curves
# REDDIT CLASSIF ELMO (16,64,256,1024)
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00089, run_name = reddit_pair_classif-16k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00357, run_name = reddit_pair_classif-64k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.01430, run_name = reddit_pair_classif-256k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.05719, run_name = reddit_pair_classif-1024k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# REDDIT CLASSIF NOELMO (16,64,256,1024)
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00089, run_name = reddit_pair_classif-16k-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00357, run_name = reddit_pair_classif-64k-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.01430, run_name = reddit_pair_classif-256k-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.05719, run_name = reddit_pair_classif-1024k-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# REDDIT S2S ELMO (16,64,256,1024)
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00089, run_name = reddit_s2s-16k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00357, run_name = reddit_s2s-64k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00357, run_name = reddit_s2s-256k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.05719, run_name = reddit_s2s-1024k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# REDDIT S2S NOELMO (16,64,256,1024)
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00089, run_name = reddit_s2s-16k-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00357, run_name = reddit_s2s-64k-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00357, run_name = reddit_s2s-256k-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.05719, run_name = reddit_s2s-1024k-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Target task learning curves
# MNLI
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.00255, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-1k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.00255, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-1k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.00255, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.01019, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-4k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.01019, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-4k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.01019, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.04074, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-16k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.04074, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-16k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.04074, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-16k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.16297, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-64k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.16297, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-64k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.16297, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-64k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch



# QQP
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.00275, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-1k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.00275, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-1k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.00275, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.01099, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-4k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.01099, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-4k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.01099, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.04397, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-16k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.04397, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-16k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.04397, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-16k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.17589, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-64k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.17589, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-64k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.17589, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-64k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch



# SST
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.01485, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-1k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.01485, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-1k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.01485, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.05939, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-4k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.05939, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-4k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.05939, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.23757, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-16k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.23757, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-16k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.23757, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-16k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.47514, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-32k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.47514, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-32k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.47514, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-32k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch



# STS-B
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.17391, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-1k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.17391, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-1k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.17391, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.69565, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-4k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.69565, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-4k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.69565, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch



# COLA
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.11695, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-1k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.11695, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-1k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.11695, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.46778, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-4k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.46778, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-4k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.46778, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch



# MRPC
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.27255, do_train = 0, eval_tasks = mrpc, do_eval = 1, run_name = lc-target-mrpc-1k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.27255, do_train = 0, eval_tasks = mrpc, do_eval = 1, run_name = lc-target-mrpc-1k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.27255, do_train = 0, eval_tasks = mrpc, do_eval = 1, run_name = lc-target-mrpc-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch



# QNLI
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.00922, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-1k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.00922, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-1k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.00922, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.03689, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-4k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.03689, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-4k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.03689, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.14755, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-16k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.14755, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-16k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.14755, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-16k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.59021, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-64k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, eval_data_fraction = 0.59021, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-64k-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = "wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded", val_interval = 10000, eval_data_fraction = 0.59021, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-64k-mtl, elmo_chars_only = 1, dec_val_scale = 250" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

