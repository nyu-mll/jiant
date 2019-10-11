# This script evaluate Transformer-XL pretrained with wt103 on blimp dataset
# Before running, set up jiant and download blimp data 


# full sentence evaluation
python main.py --config_file jiant/config/blimp/blimp_gpt2.conf --overrides="exp_name=blimp-transfo-xl,run_name=simplelm,target_tasks=blimp-simpleLM,input_module=transfo-xl-wt103"

# one prefix evaluation
python main.py --config_file jiant/config/blimp/blimp_gpt2.conf --overrides="exp_name=blimp-transfo-xl,run_name=oneprefix,target_tasks=blimp-oneprefix,input_module=transfo-xl-wt103"

# two prefix evaluation
python main.py --config_file jiant/config/blimp/blimp_gpt2.conf --overrides="exp_name=blimp-transfo-xl,run_name=twoprefix,target_tasks=blimp-twoprefix,input_module=transfo-xl-wt103"
