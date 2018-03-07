import os
import pdb

import numpy as np
from collections import defaultdict

SCRATCH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/ckpts/%s/%s/'
#SCRATCH_PREFIX = '/beegfs/aw3272/ckpts/%s/%s/'
PROJECT_NAME = 'mtl-sent-rep'
EXP_NAME = 'n_layers_01_15_18'
EXP_DIR = SCRATCH_PREFIX % (PROJECT_NAME, EXP_NAME)

TASKS = ['mnli', 'snli', 'msrp', 'sst',
         'twitter-irony', 'sts-benchmark', 'twitter', 'sts',
         'rte', 'quora', 'micro', 'macro']

OUT_FILE = "%s/summary.tex" % EXP_DIR

def extract_name_and_run(name):
    components = name.split('_')
    run_idx = int(components[-1][1:])
    return '_'.join(components[:-1]), run_idx

def extract_task(name):
    components = name.split('_')
    run_task = []
    for task in TASKS:
        if task in components:
            if '-' in task:
                run_task.append(task.split('-')[0])
            else:
                run_task.append(task)
    try:
        assert len(run_task) == 1
    except:
        pdb.set_trace()
    return run_task[0]

def extract_results(log_file):
    results = {'valid': defaultdict(dict), 'test': defaultdict(dict)}
    mode = 'none'
    with open(log_file) as fh:
        for row in fh:
            if mode == 'test':
                raw = row.split(',')
                task, scores = raw[0].split('_')[0], [s.split(':') for s in raw[1:]]
                results['test'][task] = {k:v for k, v in scores}
            elif mode == 'valid':
                raw = row.split(', ')
                if len(raw) == 1:
                    mode = 'none'
                else:
                    task, n_epochs, scores = raw[0].split('_')[0], int(raw[1]), [s.split(': ') for s in raw[2:]]
                    results['valid'][task] = {k:float(v) for k, v in scores}

            if 'VALIDATION RESULTS' in row:
                mode = 'valid'
            if 'TEST RESULTS' in row:
                mode = 'test'

    if len(results['test']) == 0:
        pdb.set_trace()
    return results

def extract_results_one(log_file, task):
    results = {'valid': defaultdict(dict), 'test': defaultdict(dict)}
    mode = 'none'
    with open(log_file) as fh:
        for row in fh:
            if mode == 'test':
                raw = row.split(',')
                metric, scores = raw[0].split('_')[0], [s.split(':') for s in raw[1:]]
                results['test'][task][metric] = float(scores[0][1])
            elif mode == 'valid':
                raw = row.split(', ')
                if len(raw) == 1:
                    mode = 'none'
                else:
                    metric, n_epochs, scores = raw[0].split('_')[0], int(raw[1]), [s.split(': ') for s in raw[2:]]
                    results['valid'][task][metric] = float(scores[0][1])

            if 'VALIDATION RESULTS' in row:
                mode = 'valid'
            if 'TEST RESULTS' in row:
                mode = 'test'

    if len(results['test']) == 0:
        pdb.set_trace()
    return results

def get_runs(exp_dir):
    runs = os.listdir(exp_dir)
    filtered_runs = [run for run in runs if 'attn' not in run and 'vocab' not in run and 'summary' not in run]
    #filtered_runs = [run for run in runs if 'vocab' not in run and 'summary' not in run]
    return filtered_runs

def latexify(out_file, results):

    def get_orders(results):
        run_order, metric_order = [], ['macro', 'micro']
        run_order += [k for k in results['valid'].keys() if k not in run_order]
        metric_order += [k for k in results['valid'][run_order[0]].keys() if k not in metric_order]
        task_order = [k for k in results['valid'][run_order[0]][metric_order[0]].keys()]
        task_order.sort()
        return run_order, metric_order, task_order

    variances = []
    with open(out_file, 'w') as fh:
        header = 'run & metric & '
        run_order, metric_order, task_order = get_orders(results)
        header += ' & '.join(task_order) + ' \\\\\n'
        header += '\\hline\n'
        fh.write(header)
        for run_name in run_order:
            for metric in metric_order:
                if not len(results['valid'][run_name][metric]):
                    continue
                res_str = '%s & %s' % (run_name, metric)
                scores = results['valid'][run_name][metric]
                for task in task_order:
                    try:
                        res_str += ' & %.3f' % np.mean(scores[task])
                        variances.append(np.var(scores[task]))
                    except:
                        pdb.set_trace()
                res_str += ' \\\\\n'
                fh.write(res_str)
    print("Average variance: %.3f" % (np.mean(variances))


# aggregated results
# structure is something like
# run name: results
#           stopping criteria : metric : scores
all_results = {'valid': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
               'test': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))}

# aggregate results
for run_dir in get_runs(EXP_DIR):
    if 'all' not in run_dir:
        task = extract_task(run_dir)
        run_results = extract_results_one(os.path.join(EXP_DIR, run_dir, 'log.log'), task)
    else:
        task = 'all'
        run_results = extract_results(os.path.join(EXP_DIR, run_dir, 'log.log'))
    run_name, run_idx = extract_name_and_run(run_dir)
    for metric, scores in run_results['valid'].items():
        metric = extract_task(metric)
        for score, value in scores.items():
            if 'loss' in score:
                continue
            score = extract_task(score)
            all_results['valid'][run_name][metric][score].append(value)
    for metric, value in run_results['test'].items():
        for score, value in scores.items():
            if 'loss' in score:
                continue
            score = extract_task(score)
            all_results['test'][run_name][metric][score].append(value)


# display results in a useful way, e.g. can copy to notebook easily
for run_name in all_results['valid'].keys():
    print('Variant: %s' % run_name)
    for metric, scores in all_results['valid'][run_name].items():
        print('\tMetric: %s' % metric)
        for score, values in scores.items():
            print('\t\t%s: %.3f' % (score, sum(values) / len(values)))
            all_results['valid'][run_name][metric][scores][score] = np.array(values)

latexify(OUT_FILE, all_results)
