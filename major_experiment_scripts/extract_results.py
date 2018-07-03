import re
import sys
import datetime


if len(sys.argv) < 2:
  print("Usage: python extract_results.py log.log")
  exit(0)

col_order = ['date', 'train_tasks', 'dropout', 'elmo', 'cola_mcc', 'sst_accuracy', 'mrpc_accuracy', 'mrpc_f1', 'sts-b_pearsonr', 'sts-b_spearmanr', 'mnli_accuracy', 'qnli_accuracy', 'rte_accuracy', 'wnli_accuracy', 'qqp_accuracy', 'qqp_f1']

cols = {c : '' for c in col_order}

today = datetime.datetime.now()
cols['date'] =  today.strftime("%m/%d/%Y")

# looking at all lines is overkill, but just in case we change the format later, 
# or if there is more junk after the eval line
results_line = None
found_eval = False
train_tasks = None
dropout = None
elmo = None

for path in sys.argv[1:]:
  with open(path) as f:
    for line in f:
      line = line.strip()

      if line == 'Evaluating...':
        found_eval = True
      else:
        if found_eval:
          assert (results_line is None), "Error! Multiple GLUE evals in this log\n"
          results_line = line.strip()
        found_eval = False

      train_m = re.match('Training model on tasks: (.*)', line)
      if train_m:
        task = train_m.groups()[0]
        if train_tasks is not None:
          assert (task == train_tasks), "Error! Multiple starts to training tasks, but tasks don't match: %s vs. %s"%(train_tasks, task)
        train_tasks = task

      do_m = re.match('"dropout": (.*),', line)
      if do_m:
        do = do_m.groups()[0]
        if dropout is not None:
          assert (dropout == do), "Error! Multiple dropouts set, but dropouts don't match: %s vs. %s"%(dropout, do)
        dropout = do

      el_m = re.match('"elmo_chars_only": (.*),', line)
      if el_m:
        el = el_m.groups()[0]
        if elmo is not None:
          assert (elmo == el), "Error! Multiple elmo flags set, but settings don't match: %s vs. %s"%(elmo, el)
        elmo = el

  cols['train_tasks'] = train_tasks
  cols['dropout'] = dropout
  cols['elmo'] = 'Y' if elmo == '0' else 'N'

  for mv in results_line.strip().split(','):
    metric, value = mv.split(':')
    cols[metric.strip()] = '%.02f'%(100*float(value.strip()))

  print(path)
  print('\t'.join(col_order))
  print('\t'.join([cols[c] for c in col_order]))

