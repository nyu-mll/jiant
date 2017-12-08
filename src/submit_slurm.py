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

#LOG_PATH="${SCRATCH_PREFIX}/ckpts/$DATE/${EXP_NAME}/info.log"
DATE = datetime.datetime.now().strftime("%m-%d")
SCRATCH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/ckpts/' + DATE + '/'
EXP_PREFIX = 'random'
GPUID = str(0)
LOAD_MODEL = str(0)
LOAD_TASKS = str(1)
LOAD_VOCAB = str(1)
LOAD_INDEX = str(1)

TASKS = ['msrp', 'rte8', 'sst', 'quora', 'snli', 'mnli']
HID_DIMS = [1024]#, 2048]
BPP_METHOD = 'percent_tr'
N_BPPS = [1000, 100, 10, 1]
BATCHES_BTW_VALIDATION = 100
LRS = [1., .1]
N_RESTARTS = 1

# Varying the validation metric for multi task
for metric in ['micro', 'macro'] + TASKS:
    #for n_bpp in N_BPPS:
    for hid_dim in HID_DIMS:
        exp_name = "%s_metric_%s" % (EXP_PREFIX, metric)
        dir_name = SCRATCH_PREFIX + exp_name
        out_file = dir_name + '/sbatch.out'
        err_file = dir_name + '/sbatch.err'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        n_bpp = 10
        bats_btw_val = n_bpp #int(BATCHES_BTW_VALIDATION / n_bpp)
        cmd = ["sbatch", "-J", exp_name, "-e", err_file, "-o", out_file,
               "run_stuff.sh", ','.join(TASKS), GPUID, exp_name,
               LOAD_MODEL, LOAD_TASKS, LOAD_VOCAB, LOAD_INDEX, DATE,
               metric, BPP_METHOD, str(n_bpp), str(bats_btw_val), str(hid_dim),
               'random']
        subprocess.call(cmd)
        print(' '.join(cmd))
        time.sleep(3)

'''
for task in TASKS:
    for hid_dim in HID_DIMS:
        for lr in LRS:
            exp_name = "%s_%s_hid_dim_%d_lr_%.3f" % (EXP_PREFIX, task, hid_dim, lr)
            dir_name = SCRATCH_PREFIX + exp_name
            out_file = dir_name + '/sbatch.out'
            err_file = dir_name + '/sbatch.err'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            cmd = ["sbatch", "-J", exp_name, "-e", err_file, "-o", out_file,
                   "run_stuff.sh", task, GPUID, exp_name,
                   LOAD_MODEL, LOAD_TASKS, LOAD_VOCAB, LOAD_INDEX, DATE,
                   str(hid_dim), str(lr)]
            subprocess.call(cmd)
            print(' '.join(cmd))
            time.sleep(3)
'''
