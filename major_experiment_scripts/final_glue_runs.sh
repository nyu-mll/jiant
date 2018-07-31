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

# Sam is running all these 18.

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


# Restore runs. Used as a workaround to a loading bug.
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sst,sts-b,qqp,wnli,rte,mnli-diagnostic\", train_tasks = \"glue,rte\", run_name = rte-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sst,sts-b,qqp,wnli,rte,mnli-diagnostic\", train_tasks = \"glue,cola\", run_name = cola-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sst,sts-b,qqp,wnli,rte,mnli-diagnostic\", train_tasks = \"glue,sst\", run_name = sst-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sst,sts-b,qqp,wnli,rte,mnli-diagnostic\", train_tasks = \"glue,wnli\", run_name = wnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sst,sts-b,qqp,wnli,rte,mnli-diagnostic\", train_tasks = \"glue,mrpc\", run_name = mrpc-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"wnli,mnli-diagnostic\", train_tasks = \"glue,mnli-alt\", mnli-alt_pair_attn = 0, run_name = mnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sst,sts-b,qqp,wnli,rte,mnli-diagnostic\", train_tasks = \"glue,sts-b-alt\", sts-b-alt_pair_attn = 0, run_name = sts-b-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sts-b,wnli,mnli-diagnostic\", train_tasks = \"glue,qnli-alt\", qnli-alt_pair_attn = 0, run_name = qnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"qnli,sst,sts-b,qqp,wnli,rte,mnli-diagnostic\", train_tasks = \"glue,qqp-alt\", qqp-alt_pair_attn = 0, run_name = qqp-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Second-phase restore runs. Used as a workaround to a loading bug.
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"cola,mrpc,qnli,mnli,mnli-diagnostic\", train_tasks = \"glue,cola\", run_name = cola-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"cola,mrpc,qnli,mnli,mnli-diagnostic\", train_tasks = \"glue,mrpc\", run_name = mrpc-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"cola,mrpc,qnli,mnli,mnli-diagnostic\", train_tasks = \"glue,sst\", run_name = sst-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"cola,mrpc,qnli,mnli,mnli-diagnostic\", train_tasks = \"glue,wnli\", run_name = wnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"cola,mrpc,qnli,mnli,mnli-diagnostic\", train_tasks = \"glue,mnli-alt\", mnli-alt_pair_attn = 0, run_name = mnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Pretraining task re-trains. Used to prevent task parameters from being shared between pretraining and target task training.
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"cola,mnli-diagnostic\", train_tasks = \"glue,cola\", run_name = cola-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sst,mnli-diagnostic\", train_tasks = \"glue,sst\", run_name = sst-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"rte,mnli-diagnostic\", train_tasks = \"glue,rte\", run_name = rte-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"wnli,mnli-diagnostic\", train_tasks = \"glue,wnli\", run_name = wnli-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"mrpc,mnli-diagnostic\", train_tasks = \"glue,mrpc\", run_name = mrpc-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch


## Random BiLSTM, no pretraining ##

# Sam is running.
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Restore runs. Used as a workaround to a loading bug.
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sts-b,wnli\", train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Second-phase restore runs. Used as a workaround to a loading bug.
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sst,mnli,qnli,qqp,rte,mrpc,mnli-diagnostic\", train_tasks = \"glue\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch


## MT ##

# Seq2seq

# Katherin is running.
JIANT_OVERRIDES="train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = bilinear, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = bilinear, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = bilinear, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = bilinear, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Seq2seq no attention

# Katherin is running.
JIANT_OVERRIDES="train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = none, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = none, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = none, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = none, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Reddit ##

# Seq2seq no attention

# Katherin is running.
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = none, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = none, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Seq2seq with attention

# Katherin is running.
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = bilinear, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, mt_attention = bilinear, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Classification

# Raghu Running

JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, run_name = reddit-class-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit_pair_classif_3.4G, run_name = reddit-class-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## LM ##

# Standard LM training
# Note: ELMo can't combine with language modeling, so there are no ELMo runs.

# Alex is running.
JIANT_OVERRIDES="train_tasks = bwb, run_name = bwb-lm-noelmo, elmo_chars_only = 1, lr = 0.001" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103, run_name = wiki103-lm-noelmo, elmo_chars_only = 1, lr = 0.001" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Seq2seq (Skip-Thought)

# Katherin is running.
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, mt_attention = bilinear, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, mt_attention = bilinear, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Seq2seq (Skip-Thought), no attention

# Katherin is running.
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, mt_attention = none, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, mt_attention = none, max_seq_len = 64, batch_size = 24" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Classification (DiscSent-style. Which is not DisSent style. Aaagh.)

# Alex is running.
JIANT_OVERRIDES="train_tasks = wiki103_classif, run_name = wiki103-cl-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103_classif, run_name = wiki103-cl-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## DisSent ##

# Alex is running.
JIANT_OVERRIDES="train_tasks = dissentwikifullbig, run_name = dissent-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = dissentwikifullbig, run_name = dissent-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## MSCOCO ##

# Sam is running.
JIANT_OVERRIDES="train_tasks = grounded, run_name = grounded-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = grounded, run_name = grounded-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Restore runs. Used as a workaround to a loading bug.
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"sst,sts-b,qqp,wnli,rte,mnli-diagnostic\", train_tasks = \"glue,grounded\", run_name = grounded-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch


## CCG (Note: For use in the NLI probing paper only) ##

# Until a fix from Alex, this needs to run in its own directory due to a vocabulary bug.
# Sam is running.
JIANT_OVERRIDES="train_tasks = ccg, exp_name = final_ccg, run_name = ccg-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = ccg, exp_name = final_ccg, run_name = ccg-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Restore runs. Used as a workaround to a loading bug.
JIANT_OVERRIDES="do_train = 0, eval_tasks = \"qnli,sst,sts-b,qqp,wnli,rte,mnli-diagnostic\", train_tasks = \"glue,ccg\",  exp_name = final_ccg, run_name = ccg-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

#### Only commands below this point may be edited after Wednesday 7/24 ####

## MTL ##

# GLUE MTL

# Alex is running
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 9000, run_name = mtl-glue-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 9000, run_name = mtl-glue-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Non-GLUE MTL
# TODO(Alex): set hyperparam for weighting tasks with decreasing metrics (e.g. ppl for LM, MT) relative to increasing metrics.
#   - wmt: wmt17_en_ru, wmt14_en_de
#   - reddit: reddit_s2s_3.4G, reddit_pair_classif_3.4G
#   - lm: wiki103, bwb
#   - skipthought: wiki103_s2s
#   - discsent: wiki103_classif
#   - dissent: dissentwikifullbig
#   - grounded: grounded

# Monster run with everything we've got.
JIANT_OVERRIDES="train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 10000, run_name = mtl-nonglue-all-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Do a run w/o LM so we can use full ELMo.
JIANT_OVERRIDES="train_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 8000, run_name = mtl-nonglue-nolm-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 8000, run_name = mtl-nonglue-nolm-noelmo, elmo_chars_only = 0, seq_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# All MTL
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola,wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 19000, run_name = mtl-alltasks-all-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 17000, run_name = mtl-alltasks-all-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 17000, run_name = mtl-alltasks-all-noelmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Target task learning curves ##

# TODO: Jan - Set up once we know which run works best on dev.
