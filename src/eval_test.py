import os
import xml.etree.ElementTree
import json
import ipdb as pdb

from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

if "cs.nyu.edu" in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
else:
    PATH_PREFIX = '/beegfs/aw3272/'

def evaluate(gold_file, pred_file, metrics=['acc'], skip_gold=1, skip_pred=1, gold_map=None):
    golds = []
    preds = []
    with open(gold_file) as gold_fh:
        for _ in range(skip_gold):
            gold_fh.readline()
        for row in gold_fh:
            targ = row.split('\t')[-1].strip()
            try:
                targ = int(targ)
            except:
                pass
            if gold_map is not None:
                targ = gold_map[targ]
            golds.append(targ)

    with open(pred_file) as pred_fh:
        for _ in range(skip_pred):
            pred_fh.readline()
        for row in pred_fh:
            targ = row.split('\t')[-1].strip()
            try:
                targ = int(targ)
            except:
                pass
            preds.append(targ)

    assert len(golds) == len(preds)
    n_exs = len(golds)
    if 'acc' in metrics:
        acc = sum([1 for gold, pred in zip(golds, preds) if gold == pred]) / len(golds)
        print("acc: %.3f" % acc)
    if 'f1' in metrics:
        f1 = f1_score(golds, preds)
        print("f1: %.3f" % f1)
    if 'matthews' in metrics:
        mcc = matthews_corrcoef(golds, preds)
        print("mcc: %.3f" % mcc)
    if "corr" in metrics:
        corr = pearsonr(golds, preds)
        print("pearson r: %.3f" % mcc)
        corr = spearmanr(golds, preds)
        print("spearman r: %.3f" % mcc)

#GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/tests/quora_test_ans.tsv'
#PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/quora_bl/elmo_r3_d500_lenc2_nhwy2_lr3e-4_do.0_clog_reg/quora_preds.tsv'
#evaluate(GOLD_FILE, PRED_FILE, gold_map=gold_map)

#GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/tests/mnli_mismatched_test_ans.tsv'
#PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/mnli_bl/elmo_attn_r7_d500_lenc3_nhwy0_lr1e-3_do.1_clog_reg/mnli_mismatched_preds.tsv'
#gold_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
#evaluate(GOLD_FILE, PRED_FILE, gold_map=gold_map)

GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/tests/squad_test_ans.tsv'
PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/squad_bl/elmo_attn_r0_d500_lenc1_nhwy1_lr1e-3_do.0_clog_reg/squad_preds.tsv'
gold_map = {'contains': 1, 'not_contain': 0}
evaluate(GOLD_FILE, PRED_FILE, gold_map=gold_map)
