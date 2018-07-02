import sys
import datetime


col_order = ['date', 'train_task', 'cola_mcc', 'sst_accuracy', 'mrpc_accuracy', 'mrpc_f1', 'sts-b_pearsonr', 'sts-b_spearmanr', 'mnli_accuracy', 'qnli_accuracy', 'rte_accuracy', 'wnli_accuracy', 'qqp_accuracy', 'qqp_f1']

cols = {c : '' for c in col_order}

today = datetime.datetime.now()
cols['date'] =  today.strftime("%m/%d/%Y")

# looking at all lines is overkill, but just in case we change the format later, 
# or if there is more junk after the eval line
results_line = None
found_eval = False
train_tasks = None
with open(sys.argv[1]) as f:
  for line in f:
    if line.strip() == 'Evaluating...':
      found_eval = True
    else:
      if found_eval:
        assert (results_line is None), "Error! Multiple GLUE evals in this log\n"
        results_line = line.strip()
      found_eval = False
    if line.startswith('Training model on tasks: '):
      task = line.strip().split(':')[1].strip()
      if train_tasks is not None:
        assert (task == train_tasks), "Error! Multiple starts to training tasks, but tasks don't match: %s vs. %s"%(train_tasks, task)
      train_tasks = task

cols['train_task'] = train_tasks
for mv in results_line.strip().split(','):
  metric, value = mv.split(':')
  cols[metric.strip()] = '%.02f'%(100*float(value.strip()))

print('\t'.join(col_order))
print('\t'.join([cols[c] for c in col_order]))

