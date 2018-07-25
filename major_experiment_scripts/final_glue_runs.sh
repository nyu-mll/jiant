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

# TODO: Berlin - Set this up so attention is not used during training on the next four pairs of runs. Use final.conf if possible. Verify that it works.

JIANT_OVERRIDES="train_tasks = mnli, run_name = mnli-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = mnli, run_name = mnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = qnli, run_name = qnli-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = qnli, run_name = qnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = qqp, run_name = qqp-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = qqp, run_name = qqp-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

JIANT_OVERRIDES="train_tasks = sts-b, run_name = sts-b-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = sts-b, run_name = sts-b-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Random BiLSTM, No Pretraining ##

JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## MT ##

# TODO: Edouard/Katherin - Can we add one non-Indo-European language?
# TODO: Edouard/Katherin/Raghu - Are there any model variants you want to try? Contrastive version?
# TODO: Edouard/Katherin/Raghu - Please add the binary classification variant, both with and without elmo

JIANT_OVERRIDES="train_tasks = wmt, run_name = wmt-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wmt, run_name = wmt-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## LM ##

# TODO: Yinghui - Set anything that you need to set to make sure we're using the correct encoder and the correct two datasets.
# Note: ELMo can't combine with language modeling, so there are no ELMo runs.

JIANT_OVERRIDES="train_tasks = bwb, run_name = bwb-lm-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = wiki103, run_name = wiki103-lm-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# TODO: Yinghui/Raghu - Add runs for skip-thought style training and contrastive training, both with and without elmo

## DisSent ##

JIANT_OVERRIDES="train_tasks = dissentwikifullbig, run_name = dissent-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = dissentwikifullbig, run_name = dissent-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## Reddit ##

# TODO: Raghu - Make sure this is set up to use the largest dataset that we're easily able to use.

JIANT_OVERRIDES="train_tasks = reddit, run_name = reddit-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = reddit, run_name = reddit-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

# TODO: Yinghui/Raghu - Add runs for skip-thought style training and contrastive training. Should follow the same pattern as LM above.

## MSCOCO ##

# TODO: Roma - Make sure this uses the dataset and objective function you want.
# TODO: Roma - Add a version that is just shapeworld, too (w/ and w/o elmo)

JIANT_OVERRIDES="train_tasks = grounded, run_name = grounded-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch
JIANT_OVERRIDES="train_tasks = grounded, run_name = grounded-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1" JIANT_CONF="config/final.conf" sbatch nyu_cilvr_cluster.sbatch

## DAE ##

# TODO: Roma - Add if ready.

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

# All MTL
# TODO: Alex - Finish once task list is final (and make sure val_interval is a 1000 x N tasks)

## Target task learning curves ##

# TODO: Jan - Set up once we know which run works best on dev.

