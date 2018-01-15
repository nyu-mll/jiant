'''simple script for submitting slurm jobs'''

import os
import pdb
import time
import random
import datetime
import subprocess

ARG2IDX = {'task':1,
           'EXP_NAME':2,
           'run_name':3,
           'SHOULD_TRAIN':4,
           'LOAD_MODEL':5,
           'LOAD_TASKS':6,
           'LOAD_VOCAB':7,
           'LOAD_INDEX':8,
           'order':9,
           'BPP_METHOD':10,
           'n_bpp':11,
           'bats_btw_val':12,
           'hid_dim':13,
           'lr':14,
           'pair_enc':15,
           'random_seed':16,
           'OPTIMIZER':17,
           'reg':18
          }

def build_args(): # TODO
    '''
    Build argument list from dictionary
    '''
    global_vars = globals()
    args = [0] * len(ARG2IDX)
    for arg, idx in ARG2IDX.items():
        args[idx - 1] = str(global_vars[arg])
    return args


SCRATCH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/ckpts/'
SCRATCH_PREFIX = '/beegfs/aw3272/ckpts/%s/%s/'
PROJECT_NAME = 'mtl-sent-rep'
EXP_NAME = 'all_tasks_01_10_18'
EXP_DIR = SCRATCH_PREFIX % (PROJECT_NAME, EXP_NAME)
GPUID = str(0)
SHOULD_TRAIN = str(1)
LOAD_MODEL = str(0)
LOAD_TASKS = str(1)
LOAD_VOCAB = str(1)
LOAD_INDEX = str(1)
N_RUNS = 3

PAIR_TASKS = ['msrp', 'rte8', 'quora', 'snli', 'mnli', 'rte', 'rte5', 'sts-benchmark']
SINGLE_TASKS = ['sst', 'twitter-irony']
ALL_TASKS = PAIR_TASKS + SINGLE_TASKS

PAIR_ENCS = ['attn'] #['simple', 'attn']
HID_DIMS = [1024]

BPP_METHOD = 'percent_tr'
N_BPPS = [10]
BATS_BTW_VALS = [10] #[100, 1000, 5000]
ORDERS = ['random'] #['small_to_large', 'large_to_small', 'random', 'random_per_pass']

OPTIMIZER = 'sgd'
LRS = [1.]
REGS = [0] #1e-5, 1e-3, 1e-1, 0]


# Varying the validation metric for multi task
# To have less nested loops: get Cartesian product of lists first?
TASKS = ['all'] + ALL_TASKS
for n_run in range(N_RUNS):
    for bats_btw_val in BATS_BTW_VALS:
        for hid_dim in HID_DIMS:
            for order in ORDERS:
                for n_bpp in N_BPPS:
                    bats_btw_val = n_bpp
                    for lr in LRS:
                        for reg in REGS:
                            for pair_enc in PAIR_ENCS:
                                for task in TASKS:
                                    run_name = "%s_%s_d%d_reg%.5f_r%d" % \
                                        (pair_enc, task, hid_dim, reg, n_run)
                                    run_dir = EXP_DIR + run_name
                                    out_file = run_dir + '/sbatch.out'
                                    err_file = run_dir + '/sbatch.err'
                                    if not os.path.exists(run_dir):
                                        os.makedirs(run_dir)
                                    random_seed = random.randint(1, 10000)
                                    args = build_args()
                                    cmd = ["sbatch", "-J", run_name,
                                            "-e", err_file, "-o", out_file,
                                           "--mem=32GB", "run_stuff.sh"] + args
                                    subprocess.call(cmd)
                                    print(' '.join(cmd))
                                    print("USED RANDOM SEED %d" % random_seed)
                                    time.sleep(30)
