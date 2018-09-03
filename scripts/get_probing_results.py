import os
import sys
import csv
import codecs

def pad(s, l=30):
    if len(s) > l:
        return s[:l]
    return s + ' '*(l - len(s))

def get_metadata(data_dir):
    meta_data = {}
    for f in os.listdir("%s/meta"%data_dir):
        run_name = f.rsplit('.', 1)[0]
        meta_data[run_name] = {}
        for row in csv.DictReader(codecs.open('%s/meta/%s'%(data_dir, f), encoding='utf-8'), delimiter='\t'):
            meta_data[run_name][row['index']] = (row['tuple_id'], row['pair_id'], row['task_desc'])
    return meta_data
 
def get_preds(data_dir, meta_data, models, binary=False): 
    accs = {}
    for model in models:
        accs[model] = {}
        for f in os.listdir('%s/%s'%(data_dir, model)):
            run_name = f.rsplit('.', 1)[0]
            accs[model][run_name] = {}
            for row in csv.DictReader(codecs.open('%s/%s/%s'%(data_dir, model, f), encoding='utf-8'), delimiter='\t'):
                idx = row['index']
                pred = int(row['prediction'])
                gold = int(row['true_label'])
                if binary:
                    pred = 0 if pred in [0, 2] else 1
                    gold = 0 if gold in [0, 2] else 1
                tid, pid, desc = meta_data[run_name][idx]
                if tid not in accs[model][run_name]:
                    accs[model][run_name][tid] = {}
                accs[model][run_name][tid][(pid, desc)] = (pred, gold)
    return accs

def compute_accs(accs):
    print("Run Name\tModel\tAcc\tCount\tTuple Acc\tTuple Count\tAcc^2")
    for model in accs:
        for run in accs[model]:
            total_acc = [0., 0.]
            tuple_acc = [0., 0.]
            by_desc = {}
            for tid in accs[model][run]:
                pair_acc = [0., 0.]
                for pid, desc in accs[model][run][tid]:
                    if desc not in by_desc:
                       by_desc[desc] = [0., 0.]
                    pred, gold = accs[model][run][tid][(pid, desc)]
                    if pred == gold:
                        total_acc[0] += 1
                        pair_acc[0] += 1
                        by_desc[desc][0] += 1
                    total_acc[1] += 1
                    pair_acc[1] += 1
                    by_desc[desc][1] += 1
                if pair_acc[0] == pair_acc[1]:
                    tuple_acc[0] += 1
                tuple_acc[1] += 1
            total_pcnt = total_acc[0]/total_acc[1]
            tuple_pcnt = tuple_acc[0]/tuple_acc[1]
            rand_tup = total_pcnt * total_pcnt
            print("%s\t%s\t%.03f\t%d\t%.03f\t%d\t%.03f"%(pad(run), pad(model),
                      total_pcnt, total_acc[1], tuple_pcnt, tuple_acc[1], rand_tup))
        print("")

if __name__ == '__main__':
    all_models = ['random', 'grounded', 'nli', 'ccg', 'lm', 'mt', 'reddit', 'shapeworld']
    data_dir = sys.argv[1]
    meta_data = get_metadata(data_dir)
    preds = get_preds(data_dir, meta_data, all_models, binary=True)
    compute_accs(preds)

