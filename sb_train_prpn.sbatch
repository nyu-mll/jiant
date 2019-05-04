#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=5000egwprpn
#SBATCH --output=/misc/vlgscratch4/BowmanGroup/pmh330/jiant-prpn-outs/prpn_outs/sl_wsj_prpn_%j.out
#SBATCH --error=/misc/vlgscratch4/BowmanGroup/pmh330/jiant-prpn-outs/prpn_outs/sl_wsj_prpn_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

run_name=$1 # specify run name wsj-0, wsj-egw, wsj-toronto, wsj-mnli
exp_name=$2 # EGWWSJPLOT_baseline, TORONTOWSJPLOT_baseline, etc
rand_seed=$3
source  activate jiant
mkdir -p /misc/vlgscratch4/BowmanGroup/pmh330/jiant-prpn-outs-prince/prpn_outs
save_dir=/misc/vlgscratch4/BowmanGroup/pmh330/jiant-prpn-outs-prince/prpn_outs

echo 'Starting main job...' >> $save_dir/out_prpn_egwwsj_5000.log

python main.py --config_file config/prpn_egw.conf  --overrides "load_model = 1,random_seed=$rand_seed,keep_all_checkpoints=1, force_include_wsj_vocabulary=1, max_word_v_size=30000, allow_reuse_of_pretraining_parameters = 1 ,patience=50, lr_patience=50,target_train_val_interval=300, max_seq_len=80,batch_size=5, exp_name = $exp_name, run_name =$run_name, sent_enc=prpn,skip_embs=0, word_embs=scratch, lr=0.003" >> $save_dir/out_prpn_egwwsj_5000.log 2>&1

