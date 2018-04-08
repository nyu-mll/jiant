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
exp_name = 'glove_no_char'

# define lots of parameters
elmo = 0

optimizer = 'sgd'
lrs = ['1e0', '5e-1', '1e-1', '5e-2']
d_hid = '1024'
drops = ['.1', '.2', '.3']
coves = [0]
n_layers_encs = ['1', '2']
n_layers_hwys = ['0', '1', '2']

bpp_method = 'percent_tr'
bpp_base = 10
val_interval = 10

n_runs = 1

for run_n in range(n_runs):
    for lr in lrs:
        for drop in drops:
            for cove in coves:
                for n_layers_enc in n_layers_encs:
                    for n_layers_hwy in n_layers_hwys:
                        if elmo:
                            mem_req = 84
                        else:
                            mem_req = 56

                        run_name = 'lr%s_do%s_nenc%s_nhwy%s_d%s_r%d' % \
                                    (lr, drop, n_layers_enc, n_layers_hwy, d_hid, run_n)
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
                                      '-t', '2-00:00', '--gres=gpu:%s:1' % gpu_type, '--mem=%dGB' % mem_req,
                                      '--mail-type=end', '--mail-user=aw3272@nyu.edu',
                                      'run_stuff.sh']
                        exp_args = ['-P', PATH_PREFIX, '-n', exp_name, '-r', run_name, '-S', seed, '-T', 'all',
                                    '-o', optimizer, '-l', lr, '-h', d_hid, '-D', drop,
                                    '-L', n_layers_enc, '-H', n_layers_hwy,
                                    '-M', bpp_method, '-B', str(bpp_base), '-V', str(val_interval),
                                    '-q', '-b', '128'] # turn off tqdm
                        if elmo:
                            exp_args.append('-e')
                        if cove:
                            exp_args.append('-c')

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
