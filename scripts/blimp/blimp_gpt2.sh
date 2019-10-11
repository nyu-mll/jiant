# This script evaluate pretrained GPT-2 large on blimp dataset
# Before running, set up jiant and download blimp data 


# full sentence evaluation
python main.py --config_file jiant/config/blimp/blimp_gpt2.conf --overrides="exp_name=blimp-gpt2,run_name=simplelm,target_tasks=blimp-simpleLM,input_module=gpt2-large"

# one prefix evaluation
python main.py --config_file jiant/config/blimp/blimp_gpt2.conf --overrides="exp_name=blimp-gpt2,run_name=oneprefix,target_tasks=blimp-oneprefix,input_module=gpt2-large"

# two prefix evaluation
python main.py --config_file jiant/config/blimp/blimp_gpt2.conf --overrides="exp_name=blimp-gpt2,run_name=twoprefix,target_tasks=blimp-twoprefix,input_module=gpt2-large"

