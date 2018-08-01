from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

#import ipdb as pdb

import nltk
nltk.data.path = ["/nfs/jsalt/share/nltk_data"] + nltk.data.path

# Install a few python packages using pip
#from w266_common import utils

import pip
import pkgutil
from pip._internal import main as pipmain
if not pkgutil.find_loader("tqdm"):
    pipmain(["install", "tqdm"])
if not pkgutil.find_loader("graphviz"):
    pipmain(["install", "graphviz"])


import nltk
# from  w266_common import treeviz
# # Monkey-patch NLTK with better Tree display that works on Cloud or other display-less server.
# print("Overriding nltk.tree.Tree pretty-printing to use custom GraphViz.")
# treeviz.monkey_patch(nltk.tree.Tree, node_style_fn=None, format='svg')

import os, sys, collections
import copy
from importlib import reload

import numpy as np
import nltk
from nltk.tree import Tree
from IPython.display import display, HTML
from tqdm import tqdm as ProgressBar

import logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# Helpers for this assignment
#from w266_common import utils, treeviz
# import part2_helpers
# import pcfg, pcfg_test
# import cky, cky_test

import time
t_0 = time.time()

#fxn_tags = ['ADV', 'NOM', 'DTV', 'LGS', 'PRD', 'PUT', 'SBJ', 'TPC', 'VOC', 'BNF', 'DIR', 'EXT', 'LOC', 'MNR', 'PRP', 'TMP', 'CLR', 'CLF', 'HLN', 'TTL']

form_function_discrepancies = ['ADV', 'NOM']
grammatical_rule = ['DTV', 'LGS', 'PRD', 'PUT', 'SBJ', 'TPC', 'VOC']
adverbials = ['BNF', 'DIR', 'EXT', 'LOC', 'MNR', 'PRP', 'TMP']
miscellaneous = ['CLR', 'CLF', 'HLN', 'TTL']

punctuations = ['-LRB-', '-RRB-', '-LCB-', '-RCB-', '-LSB-', '-RSB-']
#special_labels = ['PRP-S', 'WP-S']


# Using full ptb 
corpus = nltk.corpus.ptb

def find_depth(tree, subtree):
    treepositions = tree.treepositions()
    for indices in treepositions:
        if tree[indices] is subtree:
            return len(indices)
    raise runtime_error('something is wrong with implementation of find_depth')


#function converting Tree object to dictionary compatible with common JSON format
def sent_to_dict(sentence):
    json_d = {}

    text = ""
    for word in sentence.flatten():
        text += word + " "
    json_d["text"] = text

    max_height = sentence.height()
    for i, leaf in enumerate(sentence.subtrees(lambda t: t.height() == 2)): #modify the leafs by adding their index in the sentence
        leaf[0] = (leaf[0], str(i))
    targets = []
    for index, subtree in enumerate(sentence.subtrees()):
        assoc_words = subtree.leaves()
        assoc_words = [(i, int(j)) for i, j in assoc_words]
        assoc_words.sort(key=lambda elem: elem[1])
        tmp_tag_list = subtree.label().replace('=', '-').replace('|', '-').split('-')
        label = tmp_tag_list[0]
        if tmp_tag_list[-1].isdigit(): #Getting rid of numbers at the end of each tag
            fxn_tgs = tmp_tag_list[1:-1]
        else:
            fxn_tgs = tmp_tag_list[1:]
        #Special cases:
        if len(tmp_tag_list) > 1 and tmp_tag_list[1] == 'S': #Case when we have 'PRP-S' or 'WP-S'
            label = tmp_tag_list[0] + '-' + tmp_tag_list[1]
            fxn_tgs = tmp_tag_list[2:-1] if tmp_tag_list[-1].isdigit() else tmp_tag_list[2:]
        if subtree.label() in punctuations: #Case when we have one of the strange punctions, such as round brackets
            label, fxn_tgs = subtree.label(), []
        targets.append({"span1":[int(assoc_words[0][1]), int(assoc_words[-1][1]) + 1], "label": label, \
                        "info": {"height": subtree.height() - 1, "depth": find_depth(sentence, subtree), \
                        "form_function_discrepancies": list(set(fxn_tgs).intersection(set(form_function_discrepancies))), \
                        "grammatical_rule": list(set(fxn_tgs).intersection(set(grammatical_rule))), \
                        "adverbials": list(set(fxn_tgs).intersection(set(adverbials))), \
                        "miscellaneous": list(set(fxn_tgs).intersection(set(miscellaneous)))}})
    json_d["targets"] = targets
    
    json_d["info"] = {"source": "PTB"}
    
    return json_d

def tree_to_json(split, sent_list):
    import json

    data = {"data": []}
    num_sent = len(sent_list)
    #may want to parallelize this for loop
    for index, sentence in enumerate(sent_list):
#        print(index)
        data["data"].append(sent_to_dict(sentence))

    with open('ptb_' + split + '.json', 'w') as outfile:
        for datum in data["data"]:
            json.dump(datum, outfile)
            outfile.write("\n")


def is_null(tree):
    if tree.label() == '-NONE-':
        return True
    if (not isinstance(tree, str)):
        for i in range(len(tree)):        
            if isinstance(tree[i], str) or (not is_null(tree[i])): #I have trouble not using recursion here
                return False
        return True
    return False

def recur_delete(all_indices, deletee):
    modified_indices = []
    for index in all_indices:
        if len(index) >= len(deletee) and index[:len(deletee)] == deletee:
            continue
        else:
            modified_indices.append(index)
    return modified_indices

def prune(tree):
    tree_positions = tree.treepositions()
    null_children_indices = []

    while(len(tree_positions) > 0):
        tree_positions.sort(key=len)
        curr_tree_index = tree_positions.pop(0)
        if curr_tree_index == ():
            curr_subtree = tree
        else:
            curr_subtree = tree[curr_tree_index]
        if isinstance(curr_subtree, str):
            continue
        if curr_subtree.label() == '-NONE-':
            null_children_indices.append(curr_tree_index)
            continue
        for i in range(len(curr_subtree)):
            curr_child = curr_subtree[i]
            if isinstance(curr_child, str):
                continue
            if is_null(curr_child):
                null_children_index = tuple(list(curr_tree_index) + [i])
                null_children_indices.append(null_children_index)
                tree_positions = recur_delete(tree_positions, null_children_index)
        
    #Very hacky, perhaps there's a better way to remove branches of a tree:
    import re
    pruned_tree = re.sub( '\s+', ' ', str(tree)).strip()
    #print(null_children_indices)
    prune_keys = [str(tree[index]) for index in null_children_indices]
    prune_keys.sort(key=len, reverse=True)
    for prune_key in prune_keys: 
        pruned_tree = pruned_tree.replace(prune_key, "")
    return Tree.fromstring(pruned_tree)

if __name__ == "__main__":
    # Parsing file names to find file IDs corresponding to standard split of train, dev, and test data
    train_files, dev_files, dev_full_files, test_files = [], [], [], []
    for f in corpus.fileids():
        if f.split('/')[0] == 'BROWN':
            continue
        section = int(f.split('/')[1])
        file_num = int(f.split('/')[2][6:8])
        if section > 1 and section < 22:
            train_files.append(f)
        elif section == 22:
            dev_full_files.append(f)
            if file_num < 20:
                dev_files.append(f)
        elif section == 23:
            test_files.append(f)

    train_sent, dev_sent, dev_full_sent, test_sent = [], [], [], []
    use_full_ptb = True 
    if use_full_ptb:
        part2_helpers.verify_ptb_install()
        corpus = nltk.corpus.ptb  
        if not hasattr(corpus, '_parsed_sents'):
            print("Monkey-patching corpus reader...")
            corpus._parsed_sents = corpus.parsed_sents
            for train_file in train_files:
                train_sent += [s for s in corpus._parsed_sents(train_file)]
            for dev_file in dev_files:
                dev_sent += [s for s in corpus._parsed_sents(dev_file)]
            for test_file in test_files:
                test_sent += [s for s in corpus._parsed_sents(test_file)]
            for dev_full_file in dev_full_files:
                dev_full_sent += [s for s in corpus._parsed_sents(dev_full_file)]

    print("Converting to common JSON format...")
    print("Starting timer.")


    train_sent = [prune(sentence) for sentence in train_sent]
    dev_sent = [prune(sentence) for sentence in dev_sent]
    test_sent = [prune(sentence) for sentence in test_sent]
    dev_full_sent = [prune(sentence) for sentence in dev_full_sent]

    tree_to_json('train', train_sent)
    print("Finished generating train JSON.")
    tree_to_json('dev', dev_sent)
    print("Finished generating dev JSON.")
    tree_to_json('test', test_sent)
    print("Finished generating test JSON.")
    tree_to_json('dev.full', dev_full_sent)
    print("Finished generating dev.full JSON.")

    print("done.")
    print("Converting to JSON takes " + str(time.time() - t_0) + " seconds.")
