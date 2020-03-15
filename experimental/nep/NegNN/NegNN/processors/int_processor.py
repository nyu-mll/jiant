# -*-coding:utf-8-*-
#! /usr/bin/env python

from NegNN.reader.conll2obj import Data
from collections import Counter
from itertools import chain
from argparse import ArgumentParser
from NegNN.processors.utils import data2sents
import os
import pickle
import numpy as np
import sys
import codecs

fn_training = os.path.abspath("NegNN/data/training/sherlock_train.txt")
fn_dev = os.path.abspath("NegNN/data/dev/sherlock_dev.txt")


def load_train_dev(scope, event, lang, out_dir):
    # read data,get sentences as list of lists
    training = Data(fn_training)
    dev = Data(fn_dev)

    # get all strings
    sents, tags, tags_uni, labels, cues, scopes, lengths = data2sents(
        [training, dev], event, scope, lang
    )

    # build vocabularies
    voc, voc_inv = build_vocab(sents, tags, tags_uni, labels, lengths)

    # transform the tokens into integer indices
    words_idxs, tags_idxs, tags_uni_idxs, cues_idxs, scopes_idxs, labels_idxs = build_input_data(
        voc, sents, tags, tags_uni, cues, scopes, labels
    )

    # package
    data = package_data_train_dev(
        words_idxs,
        tags_idxs,
        tags_uni_idxs,
        cues_idxs,
        scopes_idxs,
        labels_idxs,
        voc,
        voc_inv,
        lengths,
    )

    # pickle data
    pickle_data(out_dir, data)

    return data


def load_test(fn_test, voc, scope, event, lang):
    # get test set
    test = reduce(lambda x, y: x + y, map(lambda z: Data(z), fn_test))

    # process the test set
    sents, tags, tags_uni, labels, cues, scopes, _ = data2sents(
        [test], event, scope, lang
    )

    # transform the tokens into integer indices
    words_idxs, tags_idxs, tags_uni_idxs, cues_idxs, scopes_idxs, labels_idxs = build_input_data(
        voc, sents, tags, tags_uni, cues, scopes, labels
    )

    return words_idxs, tags_idxs, tags_uni_idxs, cues_idxs, scopes_idxs, labels_idxs


def build_vocab(sents, tags, tags_uni, labels, lengths):
    def token2idx(cnt):
        return dict([(w, i) for i, w in enumerate(cnt.keys())])

    w2idxs = token2idx(Counter(chain(*sents)))
    # add <UNK> token
    w2idxs["<UNK>"] = max(w2idxs.values()) + 1
    t2idxs = token2idx(Counter(chain(*tags)))
    tuni2idxs = token2idx(Counter(chain(*tags_uni)))
    y2idxs = {"I": 0, "O": 1, "E": 2}

    voc, voc_inv = {}, {}
    voc["w2idxs"], voc_inv["idxs2w"] = w2idxs, {i: x for x, i in w2idxs.items()}
    voc["y2idxs"], voc_inv["idxs2y"] = y2idxs, {i: x for x, i in y2idxs.items()}
    voc["t2idxs"], voc_inv["idxs2t"] = t2idxs, {i: x for x, i in t2idxs.items()}
    voc["tuni2idxs"], voc_inv["idxs2tuni"] = (
        tuni2idxs,
        {x: i for x, i in tuni2idxs.items()},
    )

    return voc, voc_inv


def build_input_data(voc, sents, tags, tags_uni, cues, scopes, labels):

    words_idxs = [
        np.array(
            [
                voc["w2idxs"][w] if w in voc["w2idxs"] else voc["w2idxs"]["<UNK>"]
                for w in sent
            ],
            dtype=np.int32,
        )
        for sent in sents
    ]

    tags_idxs = [
        np.array([voc["t2idxs"][t] for t in tag_sent], dtype=np.int32)
        for tag_sent in tags
    ]
    tags_uni_idxs = [
        np.array([voc["tuni2idxs"][tu] for tu in tag_sent_uni], dtype=np.int32)
        for tag_sent_uni in tags_uni
    ]
    y_idxs = [
        np.array([voc["y2idxs"][y] for y in y_array], dtype=np.int32)
        for y_array in labels
    ]
    cues_idxs = [
        np.array([1 if c == "CUE" else 0 for c in c_array], dtype=np.int32)
        for c_array in cues
    ]
    scope_idxs = [
        np.array([1 if s == "S" else 0 for s in s_array], dtype=np.int32)
        for s_array in scopes
    ]

    return words_idxs, tags_idxs, tags_uni_idxs, cues_idxs, scope_idxs, y_idxs


def package_data_train_dev(
    sent_ind_x,
    tag_ind_x,
    tag_uni_ind_x,
    sent_ind_y,
    cues_idxs,
    scopes_idxs,
    voc,
    voc_inv,
    lengths,
):

    # vectors of words
    train_x, dev_x = (
        sent_ind_x[: lengths[0]],
        sent_ind_x[lengths[0] : lengths[0] + lengths[1]],
    )

    # vectors of POS tags
    train_tag_x, dev_tag_x = (
        tag_ind_x[: lengths[0]],
        tag_ind_x[lengths[0] : lengths[0] + lengths[1]],
    )

    # vectors of uni POS tags
    train_tag_uni_x, dev_tag_uni_x = (
        tag_uni_ind_x[: lengths[0]],
        tag_uni_ind_x[lengths[0] : lengths[0] + lengths[1]],
    )

    # vectors of y labels
    train_y, dev_y = (
        sent_ind_y[: lengths[0]],
        sent_ind_y[lengths[0] : lengths[0] + lengths[1]],
    )

    # vectors of cue info
    train_cue_info, dev_cue_info = (
        cues_idxs[: lengths[0]],
        cues_idxs[lengths[0] : lengths[0] + lengths[1]],
    )

    # vectors of scope info
    train_scope_info, dev_scope_info = (
        scopes_idxs[: lengths[0]],
        scopes_idxs[lengths[0] : lengths[0] + lengths[1]],
    )

    train_set = [
        train_x,
        train_tag_x,
        train_tag_uni_x,
        train_y,
        train_cue_info,
        train_scope_info,
    ]
    dev_set = [dev_x, dev_tag_x, dev_tag_uni_x, dev_y, dev_cue_info, dev_scope_info]

    return [train_set, dev_set, voc, voc_inv]


def pickle_data(out_dir, data):

    print("Storing data to %s..." % out_dir)
    with open(os.path.join(out_dir, "train_dev.pkl"), "wb") as f:
        pickle.dump(data, f)
    print("Data stored!")
