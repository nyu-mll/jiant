'''
Overwrite a config file by renaming args.
Require one argument: path_to_file.
'''

import sys
from src import config # use symlink from scripts to src

# Mapping - key: old name, value: new name
name_dict = {'task_patience':'lr_patience',
               'do_train': 'do_pretrain',
               'train_for_eval':'do_target_task_training',
               'do_eval': 'do_full_eval',
               'train_tasks':'pretrain_tasks',
               'eval_tasks':'target_tasks'}

path = sys.argv[1]
params = config.params_from_file(path)
for old_name, new_name in name_dict.items():
    if old_name in params:
        params[new_name] = params[old_name]
        del params[old_name]
config.write_params(params, path)
