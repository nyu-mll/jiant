import os
import glob
import sys

model_name =  sys.argv[1]
task_name = sys.argv[2]
cur_dir= sys.argv[3].strip()
print("task_name: ", task_name)

if task_name == "mnli_mismatched":
    output_path= cur_dir + '/experiments/output_dir/taskmaster_'+model_name.strip()+'_bestconfig/mnli/*/*model*.p'
else:
    output_path= cur_dir + '/scratch/pmh330/nyu-mll-jiant/experiments/output_dir/taskmaster_'+model_name.strip()+'_bestconfig/'+task_name+'/*/*model*.p'

for file_name in glob.glob(output_path):
    names = file_name.split('/')
    ckpt_name = names[-1]
    if ckpt_name =='model.p':
        continue
    config_no = names[-2].split('_')[-1]
    os.system("sbatch sb_predict_results.sbatch {} {} {} {}".format(model_name, task_name, config_no, ckpt_name))