'''
Overwrite params.conf by renaming config args.
Need one argument: path of params.conf to update.
'''

import sys

# key - old name: value - new name
name_dict = {'task_patience':'lr_patience',\
               'do_train': 'do_pretrain',\
               'train_for_eval':'do_target_task_training',\
               'do_eval': 'do_full_eval',\
               'train_tasks':'pretrain_tasks',\
               'eval_tasks':'target_tasks'}

path = sys.argv[1]
assert 'params.conf' in path, "Error: params.conf not found."
with open(path,'r+') as f:
    param = f.read()
    for k,v in name_dict.items():
        param = param.replace(k,v)
    f.seek(0)
    f.write(param)
