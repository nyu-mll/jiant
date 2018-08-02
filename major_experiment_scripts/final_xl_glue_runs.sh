# These pretraining runs explore pretraining a larger encoder than in the main experiments (final_glue_runs.sh).
# We privilege three subsets of the pretraining tasks: LM, MTL on GLUE, MTL on non-GLUE

# Ground rules:
# - There will be a separate tab in the sheet for these results. All results in the paper will come from that tab,
#     and all results in that tab must be from the runs described here.
# - All runs must be from the master branch.
# - We will not change defaults.conf, to preserve compatibility with older runs.
#     - Defaults shared with the main (non-XL) experiments will go in final.conf.
#     - Defaults between XL experiments will go in final_xl.conf.
# - You may not modify the overrides in these commands except through a reviewed pull request.
# - No pull requests will be allowed, except in cases of _mistakes_ in the commands below.
# - All results on the sheet must be reported through the script. You may not manually type anything in the main area of the sheet.
# - These commands are set up for NYU. You may, of course, modify everything outside the quotation marks to suit your machine setup.
#     Make sure you use final.conf if you do.


### LM ###

# Standard LM training
# Note: ELMo can't combine with language modeling, so there are no ELMo runs.

JIANT_OVERRIDES="train_tasks = bwb, run_name = bwb-lm-noelmo-xl, elmo_chars_only = 1, lr = 0.001" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103, run_name = wiki103-lm-noelmo-xl, elmo_chars_only = 1, lr = 0.001" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch

### MTL ###

## GLUE MTL ##

JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 9000, run_name = mtl-glue-noelmo-xl, elmo_chars_only = 1" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola\", val_interval = 9000, run_name = mtl-glue-elmo-xl, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch

## Non-GLUE MTL ##
# TODO(Alex): set hyperparam for weighting tasks with decreasing metrics (e.g. ppl for LM, MT) relative to increasing metrics.
#   - wmt: wmt17_en_ru, wmt14_en_de
#   - lm: wiki103, bwb
#   - skipthought: wiki103_s2s, reddit_s2s_3.4G
#   - discsent: wiki103_classif, reddit_pair_classif_3.4G
#   - dissent: dissentwikifullbig
#   - grounded: grounded

# Monster run with everything (non-GLUE) we've got.
JIANT_OVERRIDES="train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 10000, run_name = mtl-nonglue-all-noelmo-xl, elmo_chars_only = 1" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch

# Do a run w/o LM so we can use full ELMo.
JIANT_OVERRIDES="train_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 8000, run_name = mtl-nonglue-nolm-noelmo-xl, elmo_chars_only = 1" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 8000, run_name = mtl-nonglue-nolm-elmo-xl, elmo_chars_only = 0, seq_embs_for_skip = 1" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch

## All MTL ##
# Monster run with everything we've got.
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola,wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 19000, run_name = mtl-alltasks-all-noelmo-xl, elmo_chars_only = 1" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch

# Do a run w/o LM so we can use full ELMo.
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 17000, run_name = mtl-alltasks-nolm-noelmo-xl, elmo_chars_only = 1" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = \"mnli,mrpc,qnli,sst,sts-b,rte,wnli,qqp,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,wiki103_classif,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", val_interval = 17000, run_name = mtl-alltasks-nolm-elmo-xl, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final_xl.conf" sbatch nyu_cilvr_cluster.sbatch

