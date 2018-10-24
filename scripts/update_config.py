'''
Overwrite a config file by renaming args.
Require one argument: path_to_file.
'''

import sys

# Mapping - key: old name, value: new name
name_dict = {'task_patience':'lr_patience',\
               'do_train': 'do_pretrain',\
               'train_for_eval':'do_target_task_training',\
               'do_eval': 'do_full_eval',\
               'train_tasks':'pretrain_tasks',\
               'eval_tasks':'target_tasks'}

path = sys.argv[1]
assert '.conf' in path, "Error: .conf not found."
with open(path, 'r+') as f:
    param = f.read()
    for old_name, new_name in name_dict.items():
        # avoid substrings, won't replace first line
        param = param.replace('\n' + old_name, '\n' + new_name)
    f.seek(0)
    f.write(param)
