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
# TODO: These will need to be run in two parts because of https://github.com/jsalt18-sentence-repl/jiant/issues/290

JIANT_OVERRIDES="pretrain_tasks = cola, run_name = cola-noelmo, elmo_chars_only = 1" JIANT_CONF="config/final.conf" sbatch /misc/vlgscratch4/BowmanGroup/yp913/jiant/nyu_cilvr_cluster.sbatch

# TODO: Jan - Set up once we know which run works best on dev.
