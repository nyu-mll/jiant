#!/bin/bash

source user_config.sh

### FINAL EXPERIMENTS ###

## Preprocessing ##
#python main.py --config config/final.conf --overrides "exp_name = english-preproc, train_tasks = \"bwb,wiki103_s2s,reddit_s2s_3.4G\", eval_tasks = \"none\", max_seq_len = 64, run_name = english-preproc, do_train = 0, train_for_eval = 0, do_eval = 0, cuda = -1"
#python main.py --config config/final.conf --overrides "exp_name = mt-preproc, train_tasks = \"wmt17_en_ru,wmt14_en_de\", eval_tasks = \"none\", run_name = mt-preproc, max_seq_len = 64, do_train = 0, train_for_eval = 0, do_eval = 0, cuda = -1"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"reddit_s2s_3.4G,wiki103_s2s\", eval_tasks = \"none\", run_name = s2s-preproc, max_seq_len = 64, do_train = 0, train_for_eval = 0, do_eval = 0, cuda = -1"


## GLUE pretraining ##
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = mnli-alt, mnli-alt_pair_attn = 0, run_name = mnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 6"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = qnli-alt, qnli-alt_pair_attn = 0, run_name = qnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 5"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = qqp-alt, qqp-alt_pair_attn = 0, run_name = qqp-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 6"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, do_train = 1, train_for_eval = 0, train_tasks = cola, run_name = cola-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, do_train = 1, train_for_eval = 0, train_tasks = sst, run_name = sst-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, do_train = 1, train_for_eval = 0, train_tasks = rte, run_name = rte-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, do_train = 1, train_for_eval = 0, train_tasks = wnli, run_name = wnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, do_train = 1, train_for_eval = 0, train_tasks = mrpc, run_name = mrpc-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 5"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, do_train = 0, train_for_eval = 1, train_tasks = cola, run_name = cola-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 7"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = sts-b-alt, sts-b-alt_pair_attn = 0, run_name = sts-b-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 3"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, do_train = 0, train_for_eval = 1, train_tasks = rte, run_name = rte-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 5"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, do_train = 0, train_for_eval = 1, train_tasks = wnli, run_name = wnli-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 2"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, do_train = 0, train_for_eval = 1, train_tasks = mrpc, run_name = mrpc-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 5"

# Reruns for variance
partition1="cola,mnli,mrpc,rte,wnli"
partition2="qqp,sst"
partition3="qnli,sts-b"
partition23="qqp,qnli,sst,sts-b"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-noelmo-p1-s2, elmo_chars_only = 1, random_seed = 2222, cuda = 7"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-noelmo-p1-s3, elmo_chars_only = 1, random_seed = 3333, cuda = 1"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-noelmo-p1-s4, elmo_chars_only = 1, random_seed = 4444, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-noelmo-p1-s5, elmo_chars_only = 1, random_seed = 5555, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-elmo-p1-s2, elmo_chars_only = 0, sep_embs_for_skip = 1, random_seed = 2222"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-elmo-p1-s3, elmo_chars_only = 0, sep_embs_for_skip = 1, random_seed = 3333"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-elmo-p1-s4, elmo_chars_only = 0, sep_embs_for_skip = 1, random_seed = 4444"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = mnli-alt, eval_tasks = \"cola,mnli,mrpc,rte,wnli\", mnli-alt_pair_attn = 0, run_name = mnli-elmo-p1-s5, elmo_chars_only = 0, sep_embs_for_skip = 1, random_seed = 5555"
partition0="mnli-alt"
ckpt2="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/final-reruns/mnli-noelmo-p1-s2/model_state_main_epoch_60.best_macro.th"
ckpt3="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/final-reruns/mnli-noelmo-p1-s3/model_state_main_epoch_73.best_macro.th"
ckpt4="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/final-reruns/mnli-noelmo-p1-s4/model_state_main_epoch_64.best_macro.th"
ckpt5="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/final-reruns/mnli-noelmo-p1-s5/model_state_main_epoch_50.best_macro.th"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"${partition0}\", eval_tasks = \"${partition2}\", do_train = 0, train_for_eval = 1, do_eval = 1, mnli-alt_pair_attn = 0, run_name = mnli-noelmo-p2-s4, elmo_chars_only = 1, load_eval_checkpoint = ${ckpt4}, random_seed = 4444, cuda = 3"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"${partition0}\", eval_tasks = \"${partition2}\", do_train = 0, train_for_eval = 1, do_eval = 1, mnli-alt_pair_attn = 0, run_name = mnli-noelmo-p2-s5, elmo_chars_only = 1, load_eval_checkpoint = ${ckpt5}, random_seed = 5555, cuda = 1"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"${partition0}\", eval_tasks = \"${partition3}\", do_train = 0, train_for_eval = 1, do_eval = 1, mnli-alt_pair_attn = 0, run_name = mnli-noelmo-p3-s2, elmo_chars_only = 1, load_eval_checkpoint = ${ckpt2}, random_seed = 2222, cuda = 3"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"${partition0}\", eval_tasks = \"${partition3}\", do_train = 0, train_for_eval = 1, do_eval = 1, mnli-alt_pair_attn = 0, run_name = mnli-noelmo-p3-s3, elmo_chars_only = 1, load_eval_checkpoint = ${ckpt3}, random_seed = 3333, cuda = 3"

## Random encoder ##
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo-s3, elmo_chars_only = 1, allow_missing_task_map = 1, random_seed = 91011, cuda = 7"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-noelmo-s4, elmo_chars_only = 1, allow_missing_task_map = 1, random_seed = 121314, cuda = 7"

#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, cuda = 7"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition1}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p1-s2, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 2222, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition1}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p1-s3, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 3333, cuda = 0"

#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition2}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p2-s2, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 2222, cuda = 7"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition2}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p2-s3, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 3333, cuda = 2"

#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition3}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p3-s2, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 2222, cuda = 2"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition3}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p3-s3, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 3333, cuda = 2"

#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition1}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p1-s4, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 4444, cuda = 2"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition1}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p1-s5, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 5555, cuda = 5"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition2}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p2-s4, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 4444, cuda = 5"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition2}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p2-s5, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 5555, cuda = 5"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition3}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p3-s4, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 4444, cuda = 7"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = none, eval_tasks = \"${partition3}\", allow_untrained_encoder_parameters = 1, do_train = 0, run_name = random-elmo-p3-s5, elmo_chars_only = 0, sep_embs_for_skip = 1, allow_missing_task_map = 1, random_seed = 5555, cuda = 7"


## Reddit ##
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_pair_classif_3.4G, run_name = reddit-class-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0, cuda = 1"


## DisSent ##
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = dissentwikifullbig, run_name = dissent-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0, cuda = 3"


## Grounded ##
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = grounded, run_name = grounded-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, cuda = 4"


## Translation: no attn ##
# ELMo
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, wmt17_en_ru_s2s_attention = none, max_seq_len = 64, batch_size = 32, cuda = 5"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-noattn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, wmt14_en_de_s2s_attention = none, max_seq_len = 64, batch_size = 32, cuda = 2"
# no ELMo
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = wmt17_en_ru, run_name = wmt-en-ru-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, wmt17_en_ru_s2s_attention = none, max_seq_len = 64, batch_size = 32, cuda = 4"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = wmt14_en_de, run_name = wmt-en-de-s2s-noattn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, wmt14_en_de_s2s_attention = none, max_seq_len = 64, batch_size = 32, cuda = 7"

## S2S stuff
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = wiki103_s2s, run_name = wiki103-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, wiki103_s2s_s2s_attention = bilinear, max_seq_len = 64, cuda = N"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = wiki103_s2s, run_name = wiki103-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, wiki103_s2s_s2s_attention = bilinear, max_seq_len = 64, cuda = N"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-attn-noelmo, elmo_chars_only = 1, lr = 0.001, max_grad_norm = 1.0, reddit_s2s_3.4G_s2s_attention = bilinear, max_seq_len = 64, cuda = N"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_s2s_3.4G, run_name = reddit-s2s-attn-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, reddit_s2s_3.4G_s2s_attention = bilinear, max_seq_len = 64, cuda = 3"

## LM with 20k vocab ##
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = bwb, run_name = bwb-lm-noelmo, elmo_chars_only = 1, lr = 0.001, cuda = 1"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = wiki103, run_name = wiki103-lm-noelmo, elmo_chars_only = 1, lr = 0.001, cuda = 1"


## GLUE MTL ##
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, run_name = mtl-glue-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, do_train = 1, train_for_eval = 0, cuda = 6"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, val_interval = 9000, run_name = mtl-glue-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, do_train = 0, train_for_eval = 1, cuda = 2"


# non GLUE MTL
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 9000, run_name = mtl-nonglue-all-noelmo, elmo_chars_only = 1, dec_val_scale = 250, cuda = 1"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 7000, run_name = mtl-nonglue-nolm-noelmo, elmo_chars_only = 1, dec_val_scale = 250, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 7000, run_name = mtl-nonglue-nolm-elmo, elmo_chars_only = 0, seq_embs_for_skip = 1, dec_val_scale = 250, batch_size = 32, cuda = 3"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 9000, run_name = mtl-nonglue-all-noelmo-highlr, elmo_chars_only = 1, dec_val_scale = 250, cuda = 5, lr = .001"

# all MTL
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 18000, run_name = mtl-alltasks-all-noelmo, elmo_chars_only = 1, dec_val_scale = 250, do_train = 0, train_for_eval = 1, cuda = 1"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 16000, run_name = mtl-alltasks-nolm-noelmo, elmo_chars_only = 1, dec_val_scale = 250, do_train = 0, train_for_eval = 1, cuda = 6"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 16000, run_name = mtl-alltasks-nolm-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, dec_val_scale = 250, do_train = 0, train_for_eval = 1, cuda = 1"

partition1="qqp,rte,sst,sts-b,wnli"
partition2="cola,mnli,mrpc,qnli"
eval_ckpt="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/final-reruns/mtl-alltasks-nolm-elmo/model_state_main_epoch_34.best_macro.th"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded,${partition2}\", eval_tasks = \"${partition1}\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 16000, run_name = mtl-alltasks-nolm-elmo-p2, elmo_chars_only = 0, sep_embs_for_skip = 1, dec_val_scale = 250, do_train = 0, train_for_eval = 1, load_eval_checkpoint = ${eval_ckpt}, cuda = 6"



### Learning curves ###

## Reddit
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.05719, run_name = reddit_s2s-1024k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00357, run_name = reddit_s2s-256k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00357, run_name = reddit_s2s-64k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64, cuda = 6"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_s2s_3.4G, training_data_fraction = 0.00089, run_name = reddit_s2s-16k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, lr = 0.001, max_grad_norm = 1.0, max_seq_len = 64, cuda = 6"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00089, run_name = reddit_pair_classif-16k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0, cuda = 0"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00357, run_name = reddit_pair_classif-64k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0, cuda = 7"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.01430, run_name = reddit_pair_classif-256k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0, cuda = 2"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.05719, run_name = reddit_pair_classif-1024k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0, cuda = 2"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = reddit_pair_classif_3.4G, training_data_fraction = 0.00089, run_name = reddit_pair_classif-16k-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, pair_attn = 0, do_train = 0, cuda = 5"

## Target task learning curve
## MTL
run_dir="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/final-reruns/mtl-nonglue-all-noelmo-copy/"
model_state_file="model_state_main_epoch_176.best_macro.th"

# MNLI
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.00255, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.01019, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.04074, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-16k-mtl, elmo_chars_only = 1, dec_val_scale = 250"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.16297, do_train = 0, eval_tasks = mnli, do_eval = 1, run_name = lc-target-mnli-64k-mtl, elmo_chars_only = 1, dec_val_scale = 250, cuda = 0"

# QQP
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.00275, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.01099, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.04397, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-16k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.17589, do_train = 0, eval_tasks = qqp, do_eval = 1, run_name = lc-target-qqp-64k-mtl, elmo_chars_only = 1, dec_val_scale = 250, cuda = 2" 

# SST
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.01485, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.05939, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.23757, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-16k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.47514, do_train = 0, eval_tasks = sst, do_eval = 1, run_name = lc-target-sst-32k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 

# STS-B
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.17391, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.69565, do_train = 0, eval_tasks = sts-b, do_eval = 1, run_name = lc-target-sts-b-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 

# CoLA
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.11695, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.46778, do_train = 0, eval_tasks = cola, do_eval = 1, run_name = lc-target-cola-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 

# MRPC
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.27255, do_train = 0, eval_tasks = mrpc, do_eval = 1, run_name = lc-target-mrpc-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 

# QNLI
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.00922, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-1k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.03689, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-4k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.14755, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-16k-mtl, elmo_chars_only = 1, dec_val_scale = 250" 
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, load_eval_checkpoint = ${run_dir}/${model_state_file}, val_interval = 9000, eval_data_fraction = 0.59021, do_train = 0, eval_tasks = qnli, do_eval = 1, run_name = lc-target-qnli-64k-mtl, elmo_chars_only = 1, dec_val_scale = 250, cuda = 7" 

### Eval a model: rerun test ###
eval_cmd="do_train = 0, train_for_eval = 0, do_eval = 1, batch_size = 128, write_preds = test, write_strict_glue_format = 1"

#run_dir="/misc/vlgscratch4/BowmanGroup/awang/ckpts/mtl-sent-rep/final/mtl-glue-elmo-v2-preds/"
# We could load run specific params from run_dir, but I'm just copying the original cmd with overrides.
#python main.py --config ${run_dir}/params.conf --overrides "exp_name = final-reruns, train_tasks = bwb, run_name = bwb-lm-noelmo, elmo_chars_only = 1, lr = 0.001, ${eval_cmd}"

#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = bwb, run_name = bwb-lm-noelmo, elmo_chars_only = 1, lr = 0.001, ${eval_cmd}"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, val_interval = 7000, run_name = mtl-nonglue-nolm-elmo, elmo_chars_only = 0, seq_embs_for_skip = 1, dec_val_scale = 250, ${eval_cmd}"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,bwb,wiki103,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 18000, run_name = mtl-alltasks-all-noelmo, elmo_chars_only = 1, dec_val_scale = 250, ${eval_cmd}"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 16000, run_name = mtl-alltasks-nolm-elmo, elmo_chars_only = 0, sep_embs_for_skip = 1, dec_val_scale = 250, ${eval_cmd}"
#python main.py --config config/final.conf --overrides "exp_name = final-reruns, train_tasks = \"mnli-alt,mrpc,qnli-alt,sst,sts-b-alt,rte,wnli,qqp-alt,cola,wmt17_en_ru,wmt14_en_de,dissentwikifullbig,wiki103_s2s,reddit_s2s_3.4G,reddit_pair_classif_3.4G,grounded\", mnli-alt_pair_attn = 0, qnli-alt_pair_attn = 0, sts-b-alt_pair_attn = 0, qqp-alt_pair_attn = 0, pair_attn = 0, wmt17_en_ru_s2s_attention = none, wmt14_en_de_s2s_attention = none, reddit_s2s_3.4G_s2s_attention = bilinear, wiki103_s2s_s2s_attention = bilinear, val_interval = 16000, run_name = mtl-alltasks-nolm-elmo-p2, elmo_chars_only = 0, sep_embs_for_skip = 1, dec_val_scale = 250, ${eval_cmd}"
