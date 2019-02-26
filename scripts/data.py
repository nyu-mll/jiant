import os
import re
import cPickle
import copy

import numpy
import torch
import nltk
from nltk.corpus import ptb
from nltk import Tree
word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']

file_ids = ptb.fileids()
train_file_ids = []
valid_file_ids = []
test_file_ids = []
rest_file_ids = []

for id in file_ids:
    if 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
        train_file_ids.append(id)
    elif 'WSJ/22/WSJ_2200.MRG' <= id <= 'WSJ/22/WSJ_2299.MRG':
        valid_file_ids.append(id)
    elif 'WSJ/23/WSJ_2300.MRG' <= id <= 'WSJ/23/WSJ_2399.MRG':
        test_file_ids.append(id)
    elif 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/01/WSJ_0199.MRG' or 'WSJ/24/WSJ_2400.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
        rest_file_ids.append(id)


#data_path = '/misc/vlgscratch4/BowmanGroup/pmh330/datasets/'
#train_files = data_path + 'all_nli/all_nli_train.jsonl'
#valid_files = data_path + 'all_nli/all_nli_valid.jsonl'
#test_files_snli = data_path + 'snli_1.0/snli_1.0_test.jsonl'
test_files_mnli_match = '/scratch/am8676/exps/anhad_jiant/jiant/data/MNLI/original/multinli_1.0_dev_matched.jsonl'

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'@@UNKNOWN@@': 0}
        self.idx2word = ['@@UNKNOWN@@']
        self.word2frq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if self.word2idx.has_key(item):
            return self.word2idx[item]
        else:
            return self.word2idx['@@UNKNOWN@@']

    def rebuild_by_freq(self, thd=3):
        self.word2idx = {'@@UNKNOWN@@': 0}
        self.idx2word = ['@@UNKNOWN@@']

        for k, v in self.word2frq.iteritems():
            if v >= thd and (not k in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print 'Number of words:', len(self.idx2word)
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        dict_file_name = os.path.join(path, 'dict_nli.pkl')
        if os.path.exists(dict_file_name):
            self.dictionary = cPickle.load(open(dict_file_name, 'rb'))
            print("loading: ", dict_file_name)
        else:
            self.dictionary = Dictionary()
            #self.add_words(train_files)
            #self.add_words(valid_files)
            #self.add_words(test_files_snli)
            self.add_words(test_files_mnli_match)
            self.dictionary.rebuild_by_freq()
            cPickle.dump(self.dictionary, open(dict_file_name, 'wb'))

        
        #self.train, self.train_sens, self.train_trees = self.tokenize(train_files)
        #self.valid, self.valid_sens, self.valid_trees = self.tokenize(valid_files)
        #self.test_snli, self.test_snli_sens, self.test_snli_trees = self.tokenize(test_files_snli)
        self.test, self.test_sens, self.test_trees = self.tokenize(test_files_mnli_match)
        #self.test, self.test_sens, self.test_trees = self.tokenize(test_file_ids)

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()
                w = re.sub('[0-9]+', 'N', w)
                # if tag == 'CD':
                #     w = 'N'
                words.append(w)
        return words

    def add_words(self, file_name):
        # Add words to the dictionary
        f_in = open(file_name, 'r')
        for line in f_in:
            if line.strip() == '':
                continue 
            data = eval(line)
            sen_tree = Tree.fromstring(data['sentence1_parse'])
            words = self.filter_words(sen_tree)
            words = ['<EOS>'] + words + ['<EOS>']
            for word in words:
                self.dictionary.add_word(word)
            sen_tree = Tree.fromstring(data['sentence2_parse'])
            words = self.filter_words(sen_tree)
            words = ['<EOS>'] + words + ['<EOS>']
            for word in words:
                self.dictionary.add_word(word)
        f_in.close()

    def tokenize(self, file_name):

        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                if tree.label() in word_tags:
                    return tree.leaves()[0]
                else:
                    root = []
                    for child in tree:
                        c = tree2list(child)
                        if c != []:
                            root.append(c)
                    if len(root) > 1:
                        return root
                    elif len(root) == 1:
                        return root[0]
            return []

        sens_idx = []
        sens = []
        sentences = []
        trees = []
        f_in = open(file_name, 'r')
        for line in f_in:
            if line.strip() == '':
                continue
            data = eval(line)
            sentences = []
            sentences.append(Tree.fromstring(data['sentence1_parse']))
            sentences.append(Tree.fromstring(data['sentence2_parse']))
            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                if not words:
                    continue
                words = ['<EOS>'] + words + ['<EOS>']
                # if len(words) > 50:
                #     continue
                sens.append(words)
                idx = []
                for word in words:
                    idx.append(self.dictionary[word])
                sens_idx.append(torch.LongTensor(idx))
                trees.append(tree2list(sen_tree))
        f_in.close()       
        return sens_idx, sens, trees