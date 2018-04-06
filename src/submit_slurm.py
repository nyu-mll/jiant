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
exp_name = 'base'

# define lots of parameters

run_name = 'debug'
job_name = '%s_%s' % (run_name, exp_name)

# logging
exp_dir = '%s/ckpts/%s/%s/%s' % (PATH_PREFIX, proj_name, exp_name, run_name)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
out_file = exp_dir + '/sbatch.out'
err_file = exp_dir + '/sbatch.err'

seed = str(random.randint(1, 10000))

slurm_args = ['sbatch', '-J', job_name, '-e', err_file, '-o', out_file,
              '-t', '2-00:00', '--gres=gpu:%s:1' % gpu_type, '--mem=32GB',
              '--mail-type=end', '--mail-user=aw3272@nyu.edu',
              'run_stuff.sh']
exp_args = ['-P', PATH_PREFIX, '-n', exp_name, '-r', run_name, '-S', seed,
            '-T', 'all']

cmd = slurm_args + exp_args
print(' '.join(cmd))
subprocess.call(cmd)
time.sleep(10)

'''
- elmo has to have its own preprocessing
- make sure no non-strings
- order your for loops so as the most informative exps finish first
'''
