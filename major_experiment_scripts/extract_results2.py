import re
import sys
import datetime

# Pro tip: To copy the results of this script from the terminal in Mac OS, use command-alt-shift-c. That'll copy the tabs as tabs, not spaces.

if len(sys.argv) < 2:
  print("Usage: python extract_results.py log.log")
  exit(0)

col_order = ['date', 'train_tasks', 'dropout', 'elmo', 'cola_mcc', 'sst_accuracy', 'mrpc_accuracy', 'mrpc_f1', 'sts-b_pearsonr', 'sts-b_spearmanr', 'mnli_accuracy', 'qnli_accuracy', 'rte_accuracy', 'wnli_accuracy', 'qqp_accuracy', 'qqp_f1', 'path']

metrics = [ 'cola_mcc', 'mnli_accuracy','mrpc_acc_f1','qnli_accuracy','qqp_acc_f1','rte_accuracy','sst_accuracy','sts-b_corr','wnli_accuracy', 'micro_avg', 'macro_avg']
metircRE = '|'.join (metrics)

today = datetime.datetime.now()

# looking at all lines is overkill, but just in case we change the format later, 
# or if there is more junk after the eval line

for path in sys.argv[1:]:
  try:
    cols = {c : '' for c in col_order}
    cols['date'] =  today.strftime("%m/%d/%Y")
    cols['path'] = path
    results_line = None
    found_eval = False
    train_tasks = None
    dropout = None
    elmo = None
  
    found_val = None
    val_line = []

    with open(path) as f:
      for line in f:
        line = line.strip()
        
        if 'VALIDATION RESULTS' in line:
            found_val = True
        else:
            if found_val:
                if re.match('^('+ metircRE + '), (\d+),', line):
                
                    if len(val_line) == len(metrics):
                        val_line = []
                        print ("WARNING: Multiple GLUE validation found. Skipping all but last.")
                
                    epoch = re.findall('^('+ metircRE + '), (\d+),', line)
                    val_line += epoch
                    #print (epoch)
    
        if line == 'Evaluating...':
          found_eval = True
        else:
          if found_eval:
            # safe number to prune out lines we don't care about. we usually have at least 10 fields in those lines
            if len(line.strip().split()) > 10:
              if results_line is not None:
                print("WARNING: Multiple GLUE evals found. Skipping all but last.")
              results_line = line.strip()
              found_eval = False

        train_m = re.match('Training model on tasks: (.*)', line)
        if train_m:
          found_tasks = train_m.groups()[0]
          if train_tasks is not None and found_tasks != train_tasks:
            print("WARNING: Multiple sets of training tasks found. Skipping %s and reporting last."%(found_tasks))
          train_tasks = found_tasks

        do_m = re.match('"dropout": (.*),', line)
        if do_m:
          do = do_m.groups()[0]
          if dropout is None:
            # This is a bit of a hack: Take the first instance of dropout, which will come from the overall config.
            # Later matches will appear for model-specific configs.
            dropout = do

        el_m = re.match('"elmo_chars_only": (.*),', line)
        if el_m:
          el = el_m.groups()[0]
          if elmo is not None:
            assert (elmo == el), "Multiple elmo flags set, but settings don't match: %s vs. %s."%(elmo, el)
          elmo = el

    cols['train_tasks'] = train_tasks
    cols['dropout'] = dropout
    cols['elmo'] = 'Y' if elmo == '0' else 'N'

    assert results_line is not None, "No GLUE eval results line found. Still training?"
    for mv in results_line.strip().split(','):
      metric, value = mv.split(':')
      cols[metric.strip()] = '%.02f'%(100*float(value.strip()))
    print('\t'.join([cols[c] for c in col_order]))

    # print out validation epoch #
    assert val_line is not None, "No GLUE validation results line found. Still training?"
    val_dict = {}
    for metric,epoch in val_line:
        val_dict[metric] = epoch
    val_epoch_order = [val_dict[metric] for metric in metrics[0:9]]
    print ('Best epoch for metric: ', val_dict)
    print('Order: ', metrics[0:9])
    print('Epoch: ', '_'.join(val_epoch_order))
    # print out validation epoch #
    
  except BaseException as e:
    print("Error:", e, path)

