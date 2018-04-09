'''simple script for submitting slurm jobs'''
import os
import pdb
import time
import random
import datetime
import subprocess

if 'cs.nyu.edu' in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang'
    gpu_type = '1080ti'
else:
    PATH_PREFIX = '/beegfs/aw3272'
    gpu_type = 'p40'

# MAKE SURE TO CHANGE ME #
proj_name = 'mtl-sent-rep'
exp_name = 'glove_no_char_thin'

# model options
elmo = 0
deep_elmo = 0
if elmo:
    assert 'elmo' in exp_name

attn = 1
cove = 0

# Optimizer: momentum; adam?
# learning rate decay

optimizer = 'sgd'
lrs = ['1e0', '1e-1', '1e-2', '1e-3']
d_hids = ['512', '1024']
drops = ['.1', '.2', '.3']
classifiers = ['log_reg', 'mlp']

# momentum??
decay = '.2' #decays = ['.2', '.5']
n_layers_enc = '1'
n_layers_hwy = '0'
bpp_method = 'percent_tr'
bpp_base = 10
val_interval = 10

n_runs = 1

for run_n in range(n_runs):
    for lr in lrs:
        for drop in drops:
            for classifier in classifiers:
                for d_hid in d_hids:
                    if elmo:
                        mem_req = 56
                    else:
                        mem_req = 16

                    run_name = 'd%s_lr%s_do%s_c%s_r%d' % \
                                (d_hid, lr, drop, classifier, run_n)

                    if attn:
                        run_name = 'attn_' + run_name
                    if cove:
                        run_name = 'cove_' + run_name
                    if elmo:
                        run_name = 'elmo_' + run_name
                    job_name = '%s_%s' % (run_name, exp_name)

                    # logging
                    exp_dir = '%s/ckpts/%s/%s/%s' % (PATH_PREFIX, proj_name, exp_name, run_name)
                    if not os.path.exists(exp_dir):
                        os.makedirs(exp_dir)
                    out_file = exp_dir + '/sbatch.out'
                    err_file = exp_dir + '/sbatch.err'

                    seed = str(random.randint(1, 10000))

                    slurm_args = ['sbatch', '-J', job_name, '-e', err_file, '-o', out_file,
                                  '-t', '2-00:00', '--gres=gpu:%s:1' % gpu_type,
                                  '--mem=%dGB' % mem_req,
                                  '--mail-type=end', '--mail-user=aw3272@nyu.edu',
                                  'run_stuff.sh']
                    exp_args = ['-P', PATH_PREFIX, '-n', exp_name, '-r', run_name,
                                '-S', seed, '-T', 'all', '-C', classifier,
                                '-o', optimizer, '-l', lr, '-h', d_hid, '-D', drop,
                                '-M', bpp_method, '-B', str(bpp_base), '-V', str(val_interval),
                                '-q', '-b', '128', '-m'] # turn off tqdm
                    if elmo:
                        exp_args.append('-eg')
                        if deep_elmo:
                            exp_args.append('-d')
                    if cove:
                        exp_args.append('-c')
                    if attn:
                        exp_args.append('-p')
                        exp_args.append('attn')

                    cmd = slurm_args + exp_args
                    print(' '.join(cmd))
                    subprocess.call(cmd)
                    time.sleep(10)

''' READ ME!!
- elmo has to have its own preprocessing
- make sure no non-strings
- order your for loops so as the most informative exps finish first
- refresh your interactive sessions before launching a lot of jobs
'''
