from __future__ import division
from argparse import ArgumentParser
from collections import Counter
from itertools import izip
from operator import itemgetter
# from sklearn import metrics

import codecs
import pickle
import numpy
import scipy

parser = ArgumentParser()
parser.add_argument('-pred',help="File containing the prediction made the system in format WORD\\tGOLD_LABEL\\tSYS_PRED")
args = parser.parse_args()

WIN_LEFT = 4
WIN_RIGHT = 6

# FREQ_DICT = None

class SentenceAnalysis:
    
    def __init__(self,sent,gold,_sys):
        self.sent_obj = sent
        self.Y = gold
        self.P = _sys
        # self.cues = cues

        # self.c2idx = [i for i,v in enumerate(self.cues) if v==1]
        # self.s2idx = [i for i,v in enumerate(self.Y) if v==0]

        # self.analyze_scope() 
        # self.analyze_cue()
        # self.analyze_prediction()
        # self.analyze_words()
        # self.dummy_baseline()

#     def analyze_scope(self):
#         # len of the sentence
#         self.sent_len = len(sent)
#         # len of the scope
#         self.len_scope = len(self.s2idx)
#         # len of the scope to the right of the cue
#         # len of the scope to the left of the cue
#         if self.s2idx!=[]:
#             self.len_scope_left = len([idx for idx in self.s2idx if idx < self.c2idx[0]])
#             self.len_scope_right = len([idx for idx in self.s2idx if idx > self.c2idx[-1]])
#         else:
#             self.len_scope_left = self.len_scope_right = 0
#         # discontinuous scope (not including the cue gap)
#         lens_gaps = [self.s2idx[idx+1]-_int for idx,_int in enumerate(self.s2idx[:-1]) if self.s2idx[idx+1]-_int > 1 and self.s2idx[idx+1]!=self.s2idx[-1]]
#         self.is_disc = 1 if lens_gaps!=[] else 0
#         self.num_gaps = len(lens_gaps)
#         self.max_gap = max(lens_gaps) if lens_gaps!=[] else 0
        
    
#     def analyze_cue(self):
#         # cue
#         self.cue = ' '.join([w for i,w in enumerate(sent) if i in self.c2idx])
#         # is it prefixal?
#         self.prefixal = 1 if self.cue.endswith('-') else 0
#         # is it suffixal?
#         self.suffixal = 1 if self.cue.startswith('-') else 0 
#         # is it lexical? # is it multiword?
#         if not self.prefixal and not self.suffixal:
#             self.lexical = 1 if len(self.cue.split())==1 else 0
#             self.multiword = 1 if len(self.cue.split())>1 else 0
#         else:
#             self.lexical = 1
#             self.multiword = 1

#     def analyze_prediction(self):
#         # does it contain gap?
#         # get the spans of continuous scope in the gold, first fecth those indices where word at idx-1 and word at idx+1 are also in the scope
#         self.mid_idx = []
#         if len(self.s2idx) > 2:
#             self.mid_idx = [idx for i,idx in enumerate(self.s2idx,1) if idx-1 in self.s2idx and idx+1 in self.s2idx]
#             self.cont_pred = len([idx for idx in self.mid_idx if self.Y[idx]==self.P[idx]])/len(self.mid_idx) if self.mid_idx!=[] else 0
#         else:
#             self.cont_pred = 0
#         # overall accuracy?
#         self.accuracy = len([1 for _g,_s in izip(self.Y,self.P) if _g==_s])/len(self.Y)
#         # self.freq_word_missed = [freq_dict[self.sent_obj[i].word] for i,(_g,_s) in enumerate(zip(self.Y,self.P)) if self.sent_obj[i].word in freq_dict and _g!=_s]
#         # self.freq_word_correct = [freq_dict[self.sent_obj[i].word] for i,(_g,_s) in enumerate(zip(self.Y,self.P)) if self.sent_obj[i].word in freq_dict and _g==_s]
         
#     def analyze_words(self):
#         # how often the word has been seen as out of scope during training
#         self.oos_freqs = [oos_dict[w] for w in self.sent_obj]
#         # 1 is the word was predicted as wrong else 0
#         self.wrong_pred = [0 if p==y else 1 for p,y in izip(self.P,self.Y)]

#     def dummy_baseline(self):
#         self.dummy_pred = [1 if self.c2idx[0]-WIN_LEFT<=i<self.c2idx[0] or self.c2idx[-1]<i<=self.c2idx[-1]+WIN_RIGHT else 0 for i,w in enumerate(self.sent_obj)]

#     def test(self):
#         print 'sent: ',self.sent_obj
#         print 'sent len: ',self.sent_len
#         print 'gold labels: ',self.Y
#         print 'pred labels: ',self.P
#         print 'cues: ', self.cues
#         print 'c2idx: ',self.c2idx
#         print 'mid_idx: ',self.mid_idx
#         print 's2idx: ',self.s2idx
#         print 'len_scope: ',self.len_scope
#         print 'len_scope_R: ',self.len_scope_right
#         print 'len_scope_L: ',self.len_scope_left
#         print 'is_disc: ',self.is_disc
#         print 'num_gaps: ',self.num_gaps
#         print 'max_gap: ',self.max_gap
#         print 'cue: ',self.cue
#         print 'is_prefix: ',self.prefixal
#         print 'is_suffix: ',self.suffixal
#         print 'is_lexical: ',self.lexical
#         print 'is_multiword: ',self.multiword
#         print 'is_pred_cont: ',self.cont_pred
#         print 'accuracy: ',self.accuracy
#         print 'dummpy_pred: ',self.dummy_pred
#         # print self.oos_freqs
#         # print self.wrong_pred
#         # print 'freq word missed: ',self.freq_word_missed
#         # print 'freq word correct:',self.freq_word_correct

class Analysis:
    # calculate exact scope matching
    # and while doing it calculate accuracy (tp/len(sents))
    def __init__(self,sents):
        self.sents = sents

        # self.calculate_quartiles()
        self.exact_scope_matching()
        # self.f1_dummy_baseline()
        # self.factors_sent()
        # self.factor_words()
        # self.print_wrong_pred()

    # def calculate_quartiles(self):
    #     # quartiles for accuracy
    #     print "Mean accuracy:",sum([s.accuracy for s in self.sents])/len(self.sents)

    def exact_scope_matching(self):
        all_gold = [s.Y for s in self.sents]
        all_pred = [s.P for s in self.sents]
        tp = 0
        fn = 0
        fp = 0
        for g,p in izip(all_gold,all_pred):
            # true positives
            if g==p: tp+=1
            # false positive (no 0 in the gold but 0 in the sys)
            elif 0 not in g and 0 in p:
                fp+=1
            # false negative (0 in gold but sys doesn't full match)
            elif 0 in g and g!=p: fn+=1 
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        f_1 = (2*prec*rec)/(prec+rec)
        print "Full scope detected: ",tp,tp+fp+fn,tp/len(all_gold)
        print "Exact scope matching: ",prec,rec,f_1

    # def f1_dummy_baseline(self):
    #     dummy_pred = [d_s for s in self.sents for d_s in s.dummy_pred]
    #     golds = [g for s in self.sents for g in s.Y]
    #     assert len(dummy_pred)==len(golds)
    #     p,r,f1,s =  metrics.precision_recall_fscore_support(golds,dummy_pred)
    #     print "Dummy baseline: "
    #     print metrics.confusion_matrix(golds,dummy_pred)
    #     print p
    #     print r
    #     print f1

    # def factors_sent(self):
    #     # take all sents accuracies
    #     # [acc_s1,acc_s2,etc.]
    #     all_acc = [s.accuracy for s in self.sents]
    #     # accuracy correlates with:
    #     # SENT WISE
    #     # len of sent
    #     all_len = [s.sent_len for s in self.sents]
    #     corr_len = scipy.stats.pearsonr(all_acc,all_len)
    #     print "Corr. of %.4f with a p-value of %.4f for LEN_SENT" % (corr_len[0],corr_len[1])
    #     # presence of a disc
    #     disc = [s.is_disc for s in self.sents]
    #     corr_disc = scipy.stats.pearsonr(all_acc,disc)
    #     print "Corr. of %.4f with a p-value of %.4f for DISC." % (corr_disc[0],corr_disc[1])
    #     # len of a gap
    #     lens_gap = [s.max_gap for s in self.sents]
    #     corr_gap = scipy.stats.pearsonr(all_acc,lens_gap)
    #     print "Corr. of %.4f with a p-value of %.4f for GAP" % (corr_gap[0],corr_gap[1])
    #     # len_scope
    #     all_scope = [s.len_scope for s in self.sents]
    #     corr_scope = scipy.stats.pearsonr(all_acc,all_scope)
    #     print "Corr. of %.4f with a p-value of %.4f for SCOPE" % (corr_scope[0],corr_scope[1])
    #     # len_scope_left
    #     all_scope_l = [s.len_scope_left for s in self.sents]
    #     corr_scope_l = scipy.stats.pearsonr(all_acc,all_scope_l)
    #     print "Corr. of %.4f with a p-value of %.4f for SCOPE_LEFT" % (corr_scope_l[0],corr_scope_l[1])
    #     # len_scope_right
    #     all_scope_r = [s.len_scope_right for s in self.sents]
    #     corr_scope_r = scipy.stats.pearsonr(all_acc,all_scope_r)
    #     print "Corr. of %.4f with a p-value of %.4f for SCOPE_RIGHT" % (corr_scope_r[0],corr_scope_r[1])
    #     # lexical
    #     all_lex = [s.lexical for s in self.sents]
    #     corr_lex = scipy.stats.pearsonr(all_acc,all_lex)
    #     print "Corr. of %.4f with a p-value of %.4f for LEXICAL" % (corr_lex[0],corr_lex[1])
    #     # prefixal
    #     all_pref = [s.prefixal for s in self.sents]
    #     corr_pref = scipy.stats.pearsonr(all_acc,all_pref)
    #     print "Corr. of %.4f with a p-value of %.4f for PREFIXAL" % (corr_pref[0],corr_pref[1])
    #     # suffixal
    #     all_suff = [s.suffixal for s in self.sents]
    #     corr_suff = scipy.stats.pearsonr(all_acc,all_suff)
    #     print "Corr. of %.4f with a p-value of %.4f for SUFFIXAL" % (corr_suff[0],corr_suff[1])
    #     # multi-word
    #     all_multi_word = [s.multiword for s in self.sents]
    #     corr_mw = scipy.stats.pearsonr(all_acc,all_multi_word)
    #     print "Corr. of %.4f with a p-value of %.4f for MULTIWORD" % (corr_mw[0],corr_mw[1])

    # def factor_words(self):
    #     all_oos_freqs = [f for s in self.sents for f in s.oos_freqs]
    #     all_wrong_pred = [v for s in self.sents for v in s.wrong_pred]

    #     corr_oos_freq = scipy.stats.pearsonr(all_oos_freqs,all_wrong_pred)
    #     print "Corr. of %.4f with a p-value of %.4f for OOS_FREQ" % (corr_oos_freq[0],corr_oos_freq[1])

    # def print_wrong_pred(self):
    #     i = 1
    #     with codecs.open('is13/analyze/tmp_sents.txt','wb','utf8') as o:
    #         for s in self.sents:
    #             if s.accuracy !=1.0:
    #                 o.write("%d: ACCURACY OF SENT: %.4f\n" % (i,s.accuracy))
    #                 for w,g,p in izip(s.sent_obj,s.Y,s.P):
    #                     o.write('%s\t%s\t%s\n' % (w.ljust(20),g,p))
    #                 i+=1
    #     print "Sentence for manual analysis stored in ./tmp_sents.txt!"

def load_pred(pred_fname):
    # returns a list of list, where list[0] is the list of gold labels and list[1] the list of predictions
    all_sents = list()
    curr_words = list()
    curr_g_labels = list()
    curr_s_labels = list()

    with codecs.open(pred_fname,'rb','utf8') as f:
        for line in f:
            line = line.strip()
            if line!="":
                w,g,s = line.split()
                curr_words.append(w)
                curr_g_labels.append(int(g))
                curr_s_labels.append(int(s))
            else:
                all_sents.append([curr_words,curr_g_labels,curr_s_labels])
                curr_words = []
                curr_g_labels = []
                curr_s_labels = []
    return all_sents

# def load_cues(pkl_file):
#     with open(pkl_file,'rb') as f:
#         _,_,test_set,_ = pickle.load(f)
#         # _,val_set,_,_ = pickle.load(f)
#     _,_,_,_,cues,_ = test_set[0]
#     # _,_,_,_,cues,_ = val_set
#     return cues

# def load_dict_ofs(pkl_file):
#     with open(pkl_file,'rb') as f:
#         train_set,_,_,_dict = pickle.load(f)
#     train_lex,_,_,_,_,scopes = train_set
#     oos_dict = Counter()
#     idx2word = dict([(i,w) for w,i in _dict['words2idx'].items()])
#     for sent,sent_sc in izip(train_lex,scopes):
#         for w,s in izip(sent,sent_sc):
#             if s==1:
#                 oos_dict.update({idx2word[w]:1})

#     return oos_dict

# oos_dict = load_dict_ofs(args.pkl)

test = load_pred(args.pred)
# cues = list(load_cues(args.pkl))
# assert len(test)==len(cues)

# sents = [SentenceAnalysis(sent,gold,_sys,cue_sent) for (sent,gold,_sys),cue_sent in izip(test,cues)]

sents = [SentenceAnalysis(item[0],item[1],item[2]) for item in test]

# sents[0].test()
Analysis(sents)
