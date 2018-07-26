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

## GLUE Tasks as Pretraining ##

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

JIANT_OVERRIDES="train_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-alt-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-alt-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = qnli-alt, qnli-alt_pair_attn = 0, run_name = qnli-alt-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = qnli-alt, qnli-alt_pair_attn = 0, run_name = qnli-alt-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = qqp-alt-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = qqp-alt-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = sts-b-alt, sts-b-alt_pair_attn = 0, run_name = sts-b-alt-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = sts-b-alt, sts-b-alt_pair_attn = 0, run_name = sts-b-alt-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Random BiLSTM, No Pretraining ##

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## MT ##

JIANT_OVERRIDES="train_tasks = wmt14_en_ru_classif, run_name = wmt-noelmo, elmo_chars_only = 1, max_word_v_size = 40000, max_grad_norm=1.0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_ru_classif, run_name = wmt-noelmo, elmo_chars_only = 0, max_word_v_size = 40000, max_grad_norm=1.0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = wmt14_en_de_classif, run_name = wmt-noelmo, elmo_chars_only = 1, max_word_v_size = 40000, max_grad_norm=1.0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de_classif, run_name = wmt-noelmo, elmo_chars_only = 0, max_word_v_size = 40000, max_grad_norm=1.0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = wmt14_en_ru, run_name = wmt-noelmo, elmo_chars_only = 1, lr = 0.001, max_word_v_size = 40000, max_grad_norm=1.0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_ru, run_name = wmt-noelmo, elmo_chars_only = 0, lr = 0.001, max_word_v_size = 40000, max_grad_norm=1.0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch


JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-noelmo, elmo_chars_only = 1, lr = 0.001, max_word_v_size = 40000, max_grad_norm=1.0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt14_en_de, run_name = wmt-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_word_v_size = 40000, max_grad_norm=1.0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## LM ##

# Note: ELMo can't combine with language modeling, so there are no ELMo runs.

JIANT_OVERRIDES="train_tasks = bwb, run_name = bwb-lm-noelmo, elmo_chars_only = 1, lr = 0.001" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103, run_name = wiki103-lm-noelmo, elmo_chars_only = 1, lr = 0.001" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# TODO: Yinghui/Raghu - Add runs for skip-thought style training and contrastive training, both with and without elmo
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-noelmo, elmo_chars_only = 1, lr = 0.001" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103_s2s, run_name = wiki103-s2s-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = wiki103_classif, run_name = wiki103-cl-noelmo, elmo_chars_only = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103_classif, run_name = wiki103-cl-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## DisSent ##

JIANT_OVERRIDES="train_tasks = dissentwikifullbig, run_name = dissent-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = dissentwikifullbig, run_name = dissent-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Reddit ##

# TODO: Raghu - Make sure this is set up to use the largest dataset that we're easily able to use.

JIANT_OVERRIDES="train_tasks = reddit, run_name = reddit-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit, run_name = reddit-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# TODO: Yinghui/Raghu - Add runs for skip-thought style training and contrastive training. Should follow the same pattern as LM above.

## MSCOCO ##

JIANT_OVERRIDES="train_tasks = grounded, run_name = grounded-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = grounded, run_name = grounded-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## CCG (Note: For use in the NLI probing paper only) ##

JIANT_OVERRIDES="train_tasks = ccg, run_name = ccg-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = ccg, run_name = ccg-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

#### Only commands below this point may be edited after Wednesday 7/24 ####

## MTL ##

# GLUE MTL
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 9000, run_name = mtl-glue-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 9000, run_name = mtl-glue-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Non-GLUE MTL
# TODO: Alex - Finish once task list is final (and make sure val_interval is a 1000 x N tasks)
# TODO(Alex): set hyperparam for weighting tasks with decreasing metrics (e.g. ppl for LM, MT) relative to increasing metrics.
# AW: still waiting for confirmation that the follow tasks are good to go:
#   - wmt
#   - reddit
#   - dae
#   - grounded: double check with task to use

# Monster run with everything we've got
JIANT_OVERRIDES="train_tasks = \"wmt,bwb,wiki103,dissentwikifullbig,reddit,grounded\", val_interval = TODO, run_name = mtl-nonglue-all-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Do a run w/o LM so we can use full ELMo
JIANT_OVERRIDES="train_tasks = \"wmt,dissentwikifullbig,reddit,grounded\", val_interval = TODO, run_name = mtl-nonglue-nolm-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"wmt,dissentwikifullbig,reddit,grounded\", val_interval = TODO, run_name = mtl-nonglue-nolm-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# All MTL
# TODO: Alex - Finish once task list is final (and make sure val_interval is a 1000 x N tasks)
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola,wmt,bwb,wiki103,dissentwikifullbig,reddit,grounded\", val_interval = TODO, run_name = mtl-alltasks-all-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# Runs w/o LM so we can use full ELMo
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola,wmt,dissentwikifullbig,reddit,grounded\", val_interval = TODO, run_name = mtl-alltasks-nolm-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola,wmt,dissentwikifullbig,reddit,grounded\", val_interval = TODO, run_name = mtl-alltasks-nolm-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Target task learning curves ##

# TODO: Jan - Set up once we know which run works best on dev.
