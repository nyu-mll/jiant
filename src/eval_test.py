''' Scrappy eval script '''
import os
import json

from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

if "cs.nyu.edu" in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
else:
    PATH_PREFIX = '/beegfs/aw3272/'


def evaluate_mnli(
        pred_file,
        matched_file,
        mismatched_file,
        skip_gold=1,
        skip_pred=1,
        gold_map=None):
    m_golds = []
    with open(matched_file) as gold_fh:
        for _ in range(skip_gold):
            gold_fh.readline()
        for row in gold_fh:
            targ = row.split('\t')[-1].strip()
            try:
                targ = int(targ)
            except BaseException:
                pass
            '''
            try:
                targ = float(targ)
            except:
                pass
            '''
            if gold_map is not None:
                targ = gold_map[targ]
            m_golds.append(targ)
    mm_golds = []
    with open(mismatched_file) as gold_fh:
        for _ in range(skip_gold):
            gold_fh.readline()
        for row in gold_fh:
            targ = row.split('\t')[-1].strip()
            try:
                targ = int(targ)
            except BaseException:
                pass
            '''
            try:
                targ = float(targ)
            except:
                pass
            '''
            if gold_map is not None:
                targ = gold_map[targ]
            mm_golds.append(targ)

    preds = []
    with open(pred_file) as pred_fh:
        for _ in range(skip_pred):
            pred_fh.readline()
        for row in pred_fh:
            targ = row.split('\t')[-1].strip()
            try:
                targ = int(targ)
            except BaseException:
                pass
            try:
                targ = float(targ)
            except BaseException:
                pass
            preds.append(targ)

    assert len(m_golds) + len(mm_golds) == len(preds)
    n_m_exs = len(m_golds)
    m_preds = preds[:n_m_exs]
    mm_preds = preds[n_m_exs:]
    m_acc = sum([1 for gold, pred in zip(m_golds, m_preds) if gold == pred]) / len(m_golds)
    print("matched acc: %.3f" % m_acc)
    mm_acc = sum([1 for gold, pred in zip(mm_golds, mm_preds) if gold == pred]) / len(mm_golds)
    print("mismatched acc: %.3f" % mm_acc)


def evaluate(gold_file, pred_file, metrics=['acc'], skip_gold=1, skip_pred=1, gold_map=None):
    golds = []
    preds = []
    with open(gold_file) as gold_fh:
        for _ in range(skip_gold):
            gold_fh.readline()
        for row in gold_fh:
            targ = row.strip().split('\t')[-1]
            try:
                targ = int(targ)
            except BaseException:
                pass
            '''
            try:
                targ = float(targ)
            except:
                pass
            '''
            if gold_map is not None:
                targ = gold_map[targ]
            golds.append(targ)

    with open(pred_file) as pred_fh:
        for _ in range(skip_pred):
            pred_fh.readline()
        for row in pred_fh:
            targ = row.strip().split('\t')[-1]
            try:
                targ = int(targ)
            except BaseException:
                pass
            preds.append(targ)

    assert len(golds) == len(preds)
    n_exs = len(golds)
    if 'acc' in metrics:
        acc = sum([1 for gold, pred in zip(golds, preds) if gold == pred]) / float(len(golds))
        print("acc: %.3f" % acc)
    if 'f1' in metrics:
        f1 = f1_score(golds, preds)
        print("f1: %.3f" % f1)
    if 'matthews' in metrics:
        mcc = matthews_corrcoef(golds, preds)
        print("mcc: %.3f" % mcc)
    if "corr" in metrics:
        corr = pearsonr(golds, preds)[0]
        print("pearson r: %.3f" % corr)
        corr = spearmanr(golds, preds)[0]
        print("spearman r: %.3f" % corr)


def evaluate_sts(gold_file, pred_file, skip_gold=1, skip_pred=1, gold_map=None):
    golds = []
    preds = []
    with open(gold_file) as gold_fh:
        for _ in range(skip_gold):
            gold_fh.readline()
        for row in gold_fh:
            targ = row.split('\t')[-1].strip()
            try:
                targ = float(targ)
            except BaseException:
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
                targ = float(targ)
            except BaseException:
                pass
            preds.append(targ)

    assert len(golds) == len(preds)
    n_exs = len(golds)
    corr = pearsonr(golds, preds)[0]
    print("pearson r: %.3f" % corr)
    corr = spearmanr(golds, preds)[0]
    print("spearman r: %.3f" % corr)


codebase = 'mtl-sent-rep'
run_n = 1
exp = 'base_attn'
if 'elmo' in exp:
    exp_dir = 'elmo_no_glove_v3'
else:
    exp_dir = 'glove_v3'
run_dir = 'r%d_%s_bpp1_vi10000_d1500_lenc2_nhwy0_adam_lr1e-3_decay.2_p5_tp1_maxscale_do0.2_cmlp' % (
    run_n, exp)

#codebase = 'SentEval'
#exp_dir = 'infersent'
#run_dir = 'r8_benchmark_v3'

tasks = 'acceptability'

if 'mnli' in tasks or 'benchmark' in tasks:
    print('MNLI matched')
    M_GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/tests/mnli_matched_test_ans.tsv'
    MM_GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/tests/mnli_mismatched_test_ans.tsv'
    PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/%s/mnli_preds.tsv' % (codebase, exp_dir, run_dir)
    #PRED_FILE = PATH_PREFIX + 'ckpts/SentEval/infersent/debug_preds/MNLI_preds.tsv'
    gold_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
    evaluate_mnli(PRED_FILE, M_GOLD_FILE, MM_GOLD_FILE, gold_map=gold_map)

if 'msrp' in tasks or 'benchmark' in tasks:
    print('MSRP')
    GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations-old/tests/msrp_test_ans.tsv'
    #PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/%s/%s/msrp_preds.tsv' % (exp_dir, run_dir)
    PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/%s/MRPC.tsv' % (codebase, exp_dir, run_dir)
    #PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/%s/msrp_preds.tsv' % (codebase, exp_dir, run_dir)
    evaluate(GOLD_FILE, PRED_FILE, metrics=['acc', 'f1'])

if 'quora' in tasks or 'benchmark' in tasks:
    print('QQP')
    GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/tests/quora_test_ans.tsv'
    PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/%s/%s/quora_preds.tsv' % (exp_dir, run_dir)
    gold_map = {'contains': 1, 'not_contain': 0}
    evaluate(GOLD_FILE, PRED_FILE, metrics=['acc', 'f1'])  # , gold_map=gold_map)

# RTE
if 'rte' in tasks or 'benchmark' in tasks:
    print('RTE')
    GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations-old/tests/rte_test_ans.tsv'
    #PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/%s/%s/rte_preds.tsv' % (exp_dir, run_dir)
    PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/%s/RTE.tsv' % (codebase, exp_dir, run_dir)
    PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/RTE.tsv' % (codebase, exp_dir)
    evaluate(GOLD_FILE, PRED_FILE)  # , gold_map=gold_map)

# SQuAD
if 'squad' in tasks or 'benchmark' in tasks:
    print('SQuAD')
    GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations-old/tests/squad_test_ans.tsv'
    PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/%s/%s/squad_preds.tsv' % (exp_dir, run_dir)
    PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/%s/SQuAD.tsv' % (codebase, exp_dir, run_dir)
    #gold_map = {'contains': 1, 'not_contain': 0, 'entailment': 1, 'not_entailment': 0}
    gold_map = {'contains': 'entailment', 'not_contain': 'not_entailment'}
    evaluate(GOLD_FILE, PRED_FILE, gold_map=gold_map)

# SST
if 'sst' in tasks or 'benchmark' in tasks:
    print('SST')
    GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/tests/sst_binary_test_ans.tsv'
    #PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/%s/%s/sst_preds.tsv' % (exp_dir, run_dir)
    PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/%s/%s/sst.tsv' % (exp_dir, run_dir)
    evaluate(GOLD_FILE, PRED_FILE)

# STS-B
if 'sts' in tasks or 'benchmark' in tasks:
    print('STS-B')
    GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/tests/sts_benchmark_test_ans.tsv'
    #PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/%s/sts-b_preds.tsv' % (codebase, exp_dir, run_dir)
    PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/%s/STSBenchmark.tsv' % (codebase, exp_dir, run_dir)
    evaluate_sts(GOLD_FILE, PRED_FILE)

# Warstadt
if 'acceptability' in tasks or 'benchmark' in tasks:
    print("Warstadt Acceptability")
    GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/CoLA/test_ans.tsv'
    PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/%s/acceptability.tsv' % (codebase, exp_dir, run_dir)
    #PRED_FILE = PATH_PREFIX + 'ckpts/%s/%s/%s/Warstadt.tsv' % (codebase, exp_dir, run_dir)
    evaluate(GOLD_FILE, PRED_FILE, metrics=['matthews'])

# WNLI
if 'wnli' in tasks or 'benchmark' in tasks:
    print('WNLI')
    GOLD_FILE = PATH_PREFIX + 'processed_data/mtl-sentence-representations/tests/wnli_test_ans.tsv'
    PRED_FILE = PATH_PREFIX + 'ckpts/mtl-sent-rep/%s/%s/wnli_preds.tsv' % (exp_dir, run_dir)
    evaluate(GOLD_FILE, PRED_FILE)
