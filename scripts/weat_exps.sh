#!/bin/bash

source user_config.sh

# Debug
#python -m ipdb extract_repr.py --config config/bias.conf --overrides "target_tasks = weat1, run_name = cove, word_embs = glove, cove = 1, elmo = 0"
#python extract_repr.py --config config/bias.conf --overrides "target_tasks = weat1-openai, exp_name = sentbias-openai, run_name = openai, word_embs = none, elmo = 0, openai_transformer = 1, sent_enc = \"null\", skip_embs = 1, sep_embs_for_skip = 1, allow_missing_task_map = 1"

# CoVe
#python extract_repr.py --config config/bias.conf --overrides "target_tasks = \"weat1,weat2,weat3,weat4\", run_name = cove, word_embs = glove, cove = 1, elmo = 0, sent_enc = \"null\", skip_embs = 1, sep_embs_for_skip = 1, allow_missing_task_map = 1, combine_method = max"
#python extract_repr.py --config config/bias.conf --overrides "target_tasks = \"sent-weat1,sent-weat2,sent-weat3,sent-weat4\", run_name = cove, word_embs = glove, cove = 1, elmo = 0, sent_enc = \"null\", skip_embs = 1, sep_embs_for_skip = 1, allow_missing_task_map = 1, combine_method = max"
#python extract_repr.py --config config/bias.conf --overrides "target_tasks = \"weat5,weat5b,weat6,weat6b,weat7,weat7b,weat8,weat8b,weat9,weat10\", run_name = cove, word_embs = glove, cove = 1, elmo = 0, sent_enc = \"null\", skip_embs = 1, sep_embs_for_skip = 1, allow_missing_task_map = 1, combine_method = max"

# OpenAI GPT
# NOTE: make sure to retokenize the tests first!
#python extract_repr.py --config config/bias.conf --overrides "target_tasks = \"weat1-openai,weat2-openai,weat3-openai,weat4-openai\", exp_name = sentbias-openai, run_name = openai, word_embs = none, elmo = 0, openai_transformer = 1, sent_enc = \"null\", skip_embs = 1, sep_embs_for_skip = 1, allow_missing_task_map = 1, combine_method = last"
#python extract_repr.py --config config/bias.conf --overrides "target_tasks = \"sent-weat1-openai,sent-weat2-openai,sent-weat3-openai,sent-weat4-openai\", exp_name = sentbias-openai, run_name = openai-sweat, word_embs = none, elmo = 0, openai_transformer = 1, sent_enc = \"null\", skip_embs = 1, sep_embs_for_skip = 1, allow_missing_task_map = 1, combine_method = last"
python extract_repr.py --config config/bias.conf --overrides "target_tasks = \"weat5-openai,weat5b-openai,weat6-openai,weat6b-openai,weat7-openai,weat7b-openai,weat8-openai,weat8b-openai,weat9-openai,weat10-openai\", exp_name = sentbias-openai, run_name = openai, word_embs = none, elmo = 0, openai_transformer = 1, sent_enc = \"null\", skip_embs = 1, sep_embs_for_skip = 1, allow_missing_task_map = 1, combine_method = last"
