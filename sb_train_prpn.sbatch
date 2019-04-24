#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=egwwsjprpn
#SBATCH --output=/beegfs/pmh330/jiant-experiments/prpn_outs/sl_egwwsj_prpn_%j.out
#SBATCH --error=/beegfs/pmh330/jiant-experiments/prpn_outs/sl_egwwsj_prpn_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=6
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

run_name=$1 # specify run name wsj-0, wsj-egw, wsj-toronto, wsj-mnli
exp_name=$2 # EGWWSJPLOT_baseline, TORONTOWSJPLOT_baseline, etc
source  activate jiant
mkdir -p /beegfs/pmh330/jiant-experiments/prpn_outs
save_dir=/beegfs/pmh330/jiant-experiments/prpn_outs

echo 'Starting main job...' >> $save_dir/out_prpn_egwwsj.log

python main.py --config_file /beegfs/pmh330/jiant/config/prpn.conf  --overrides "keep_all_checkpoints=1, force_include_wsj_vocabulary=1, max_word_v_size=30000, allow_reuse_of_pretraining_parameters = 1 ,patience=50, lr_patience=50,eval_val_interval=300, max_seq_len=80,batch_size=5, exp_name = $exp_name, run_name =$run_name, sent_enc=prpn,skip_embs=0, word_embs=scratch, lr=0.003" >> $save_dir/out_prpn_egwwsj.log 2>&1

exit
