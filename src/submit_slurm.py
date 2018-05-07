'''simple script for submitting slurm jobs'''
import os
import pdb
import time
import random
import datetime
import subprocess

if 'cs.nyu.edu' in os.uname()[1] or 'dgx' in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang'
    gpu_type = '1080ti'
else:
    PATH_PREFIX = '/beegfs/aw3272'
    gpu_type = 'p40' # should be p100 or p40

# MAKE SURE TO CHANGE ME #
proj_name = 'mtl-sent-rep'
rand_search = 0
n_runs = 1

# embedding stuff
elmo = 0
deep_elmo = 0
if elmo:
    exp_name = 'elmo_no_glove_v3'
else:
    exp_name = 'glove_v3'
cove = 0
glove = 0

# model parameters
d_hids = ['500', '1000', '1500', '2000']
n_enc_layers = ['1', '2', '3']
n_hwy_layers = ['0', '1', '2']
drops = ['0.0', '0.1', '0.2', '0.3']
classifiers = ['log_reg', 'mlp']
attn = 1

# optimization settings
optimizers = ['sgd', 'adam']
lrs = ['1e0', '1e-1']#, '1e-2', '1e-3']
decays = ['.2', '.5']

# multi task training settings
bpp_method = 'percent_tr'
bpps = [1]
val_intervals = [10000]
scales = ['none'] #['max', 'min']
weighting_method = 'proportional'

###### BEST ######

# best model settings
best_d_hid = '1500'
best_n_enc_layer = '2'
best_n_hwy_layer = '0'
best_drop = '0.2'
best_classifier = 'mlp'

# best optimizer settings
best_optimizer = 'adam'
best_lr = '1e-3'
best_lr_decay = '.2'
best_task_patience = 0
best_patience = '5'

# best multi task settings
best_bpp = 1
best_val_interval = 10000
best_scale = 'max'
best_weighting_method = 'proportional'

for run_n in range(n_runs):
    if rand_search:
        d_hid = random.choice(d_hids)
        n_enc_layer = random.choice(n_enc_layers)
        n_hwy_layer = random.choice(n_hwy_layers)
        drop = random.choice(drops)
        classifier = random.choice(classifiers)
        lr = random.choice(lrs)
    else:
        d_hid = best_d_hid
        n_enc_layer = best_n_enc_layer
        n_hwy_layer = best_n_hwy_layer
        drop = best_drop
        classifier = best_classifier

        optimizer = best_optimizer
        lr = best_lr
        lr_decay = best_lr_decay
        task_patience = best_task_patience
        patience = best_patience

        bpp = best_bpp
        val_interval = best_val_interval
        scale = best_scale
        weighting_method = best_weighting_method

    if elmo:
        mem_req = 64
    else:
        mem_req = 16

    run_name = 'bpp%d_vi%d_d%s_lenc%s_nhwy%s_%s_lr%s_decay%s_p%s_tp%s_%sscale_do%s_c%s' % \
                (bpp, val_interval, d_hid, n_enc_layer, n_hwy_layer, optimizer,
                 lr, lr_decay, patience, str(task_patience + 1), scale, drop, classifier)
    if attn:
        run_name = 'attn_' + run_name
    if cove:
        run_name = 'cove_' + run_name
    if elmo:
        run_name = 'elmo_' + run_name
    if not glove:
        run_name = 'learned_' + run_name
    if not attn and not cove and not elmo and not rand_search and glove:
        run_name = 'base_' + run_name
    run_name = ("r%d_" % run_n) + run_name
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
                '-L', n_enc_layer, '-H', n_hwy_layer,
                '-M', bpp_method, '-B', str(bpp), '-V', str(val_interval),
                '-y', lr_decay, '-K', str(task_patience), '-p', patience,
                '-W', weighting_method, '-s', scale,
                '-q', '-m'] # turn off tqdm

    exp_args.append('-b')
    if d_hid == '2000' or 'n_enc_layer' == '3':
        exp_args.append('64')
    else:
        exp_args.append('128')

    if elmo:
        exp_args.append('-eg')
        if deep_elmo:
            exp_args.append('-d')
    if not glove:
        exp_args.append('-G')
    if cove:
        exp_args.append('-c')
    if attn:
        exp_args.append('-E')
        exp_args.append('attn')

    cmd = slurm_args + exp_args
    print(' '.join(cmd))
    subprocess.call(cmd)
    time.sleep(10)
