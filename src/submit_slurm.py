'''simple script for submitting slurm jobs'''

import os
import pdb
import time
import subprocess
import datetime

ARG2IDX = {'tasks':1, # this is old
           'gpuid':2,
           'exp_name':3,
           'SHOULD_TRAIN':4,
           'LOAD_MODEL':5,
           'LOAD_TASKS':6,
           'LOAD_VOCAB':7,
           'LOAD_INDEX':8,
           'TASK_ORDERING':9,
           'BPP_METHOD':10,
           'N_BPPS':11,
           'BATCHES_BTW_VALIDATION':12,
           'HID_DIM':13,
           'LR':14,
           'PAIR_ENC':15,
           'DATE':16
          }

DATE = datetime.datetime.now().strftime("%m-%d")
DATE = "12-15"
SCRATCH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/ckpts/' + DATE + '/'
SCRATCH_PREFIX = '/beegfs/aw3272/ckpts/' + DATE + '/'
EXP_PREFIX = 'fixed'
GPUID = str(0)
SHOULD_TRAIN = str(1)
LOAD_MODEL = str(0)
LOAD_TASKS = str(1)
LOAD_VOCAB = str(1)
LOAD_INDEX = str(1)

PAIR_TASKS = ['msrp', 'rte8', 'quora', 'snli', 'mnli', 'rte', 'sts-benchmark']
SINGLE_TASKS = ['sst', 'twitter-irony']
TASKS = PAIR_TASKS + SINGLE_TASKS
HID_DIMS = [1024]
BPP_METHOD = 'fixed'
N_BPPS = [1]
BATS_BTW_VALS = [100, 1000, 5000]
LRS = [1.]
N_RUNS = 5

# Varying the validation metric for multi task
for n_bpp in N_BPPS:
    for bats_btw_val in BATS_BTW_VALS:
        for hid_dim in HID_DIMS:
            for run_idx in range(N_RUNS):
                lr = LRS[0]
                exp_name = "%s_r%d_dim_%d_bbv_%d_lr_%.3f" % (EXP_PREFIX, run_idx, hid_dim, bats_btw_val, lr)
                dir_name = SCRATCH_PREFIX + exp_name
                out_file = dir_name + '/sbatch.out'
                err_file = dir_name + '/sbatch.err'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                #bats_btw_val = n_bpp
                cmd = ["sbatch", "-J", exp_name, "-e", err_file, "-o", out_file,
                       "--mem=16GB",
                       "run_stuff.sh", ','.join(TASKS), GPUID, exp_name,
                       SHOULD_TRAIN, LOAD_MODEL, LOAD_TASKS, LOAD_VOCAB, LOAD_INDEX,
                       'small_to_large', BPP_METHOD, str(n_bpp), str(bats_btw_val), str(hid_dim), str(lr),
                       'simple', DATE]
                subprocess.call(cmd)
                print(' '.join(cmd))
                time.sleep(10)
'''

# Single task models
for hid_dim in HID_DIMS:
    for lr in LRS:
        for task in TASKS: #PAIR_TASKS:
            exp_name = "%s_%s_hid_dim_%d" % (task, EXP_PREFIX, hid_dim)
            dir_name = SCRATCH_PREFIX + exp_name
            out_file = dir_name + '/sbatch.out'
            err_file = dir_name + '/sbatch.err'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            n_bpp = bats_btw_val = 1
            bpp_method = 'percent_tr'
            task_ordering = 'small_to_large'
            pair_encoder = 'simple'
            metric = 'micro'
            cmd = ["sbatch", "-J", exp_name, "-e", err_file, "-o", out_file,
                   "--mem=16GB",
                   "run_stuff.sh", task, GPUID, exp_name,
                   LOAD_MODEL, LOAD_TASKS, LOAD_VOCAB, LOAD_INDEX, DATE,
                   metric, bpp_method, str(n_bpp), str(bats_btw_val), str(hid_dim),
                   str(lr), pair_encoder]
            subprocess.call(cmd)
            print(' '.join(cmd))
            time.sleep(5)
'''
