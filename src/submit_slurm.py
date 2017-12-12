'''simple script for submitting slurm jobs'''

import os
import pdb
import time
import subprocess
import datetime

'''
Make sure to change
- experiment prefix
'''

ARG2IDX = {'tasks':1,
           'gpuid':2,
           'exp_name':3,
           'LOAD_MODEL':4,
           'LOAD_TASKS':5,
           'LOAD_VOCAB':6,
           'LOAD_INDEX':7,
           'DATE':8,
           'VAL_METRIC':9,
           'BPP_METHOD':10,
           'N_BPPS':11,
           'BATCHES_BTW_VALIDATION':12
          }

#LOG_PATH="${SCRATCH_PREFIX}/ckpts/$DATE/${EXP_NAME}/info.log"
DATE = datetime.datetime.now().strftime("%m-%d")
SCRATCH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/ckpts/' + DATE + '/'
SCRATCH_PREFIX = '/beegfs/aw3272/ckpts/' + DATE + '/'
EXP_PREFIX = 'bidaf'
GPUID = str(0)
LOAD_MODEL = str(0)
LOAD_TASKS = str(1)
LOAD_VOCAB = str(1)
LOAD_INDEX = str(0)

PAIR_TASKS = ['msrp', 'rte8', 'quora', 'snli', 'mnli']
SINGLE_TASKS = ['sst']
TASKS = PAIR_TASKS + SINGLE_TASKS
HID_DIMS = [256, 512, 1024]
BPP_METHOD = 'percent_tr'
N_BPPS = [1000, 100, 10, 1]
BATCHES_BTW_VALIDATION = 100
LRS = [.01, .03, .3]
N_RESTARTS = 10

# Varying the validation metric for multi task
'''
for metric in ['micro']: #['micro', 'macro'] + TASKS:
    #for n_bpp in N_BPPS:
    for hid_dim in HID_DIMS:
        exp_name = "%s_%d_%s" % (EXP_PREFIX, hid_dim, metric)
        dir_name = SCRATCH_PREFIX + exp_name
        out_file = dir_name + '/sbatch.out'
        err_file = dir_name + '/sbatch.err'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        n_bpp = 10
        bats_btw_val = n_bpp #int(BATCHES_BTW_VALIDATION / n_bpp)
        cmd = ["sbatch", "-J", exp_name, "-e", err_file, "-o", out_file,
               "--mem=24GB",
               "run_stuff.sh", ','.join(TASKS), GPUID, exp_name,
               LOAD_MODEL, LOAD_TASKS, LOAD_VOCAB, LOAD_INDEX, DATE,
               metric, BPP_METHOD, str(n_bpp), str(bats_btw_val), str(hid_dim)]
        subprocess.call(cmd)
        print(' '.join(cmd))
        time.sleep(60)
'''
# Single task models
for hid_dim in HID_DIMS:
    for lr in LRS:
        for task in PAIR_TASKS:
            exp_name = "%s_%s_hid_dim_%d_lr_%.3f" % (task, EXP_PREFIX, hid_dim, lr)
            dir_name = SCRATCH_PREFIX + exp_name
            out_file = dir_name + '/sbatch.out'
            err_file = dir_name + '/sbatch.err'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            n_bpp = bats_btw_val = 1
            bpp_method = 'percent_tr'
            task_ordering = 'small_to_large'
            pair_encoder = 'bidaf'
            metric = 'micro'
            cmd = ["sbatch", "-J", exp_name, "-e", err_file, "-o", out_file,
                   "--mem=8GB",
                   "run_stuff.sh", task, GPUID, exp_name,
                   LOAD_MODEL, LOAD_TASKS, LOAD_VOCAB, LOAD_INDEX, DATE,
                   metric, bpp_method, str(n_bpp), str(bats_btw_val), str(hid_dim),
                   task_ordering, pair_encoder, str(lr)]
            subprocess.call(cmd)
            print(' '.join(cmd))
            time.sleep(5)
