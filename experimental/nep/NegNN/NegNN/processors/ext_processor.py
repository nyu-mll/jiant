# -*-coding:utf-8-*-
#! /usr/bin/env python
#
from NegNN.reader.conll2obj import Data
from NegNN.processors.utils import data2sents
import numpy as np
import os

fn_training = os.path.abspath("NegNN/data/training/sherlock_train.txt")
fn_dev = os.path.abspath("NegNN/data/dev/sherlock_dev.txt")


def load_words():
    w_emb = np.load(
        os.path.abspath("NegNN/w2v/words_50/VectorModel-we_lm_sh_data_50.data.syn0.npy")
    )
    idxs2w_list = np.load(os.path.abspath("NegNN/w2v/words_50/index2word.npy"))
    pre_w2idxs = dict([(w, i) for i, w in enumerate(idxs2w_list)])
    pre_idxs2w = dict([(v, k) for k, v in pre_w2idxs.items()])

    return w_emb, pre_w2idxs, pre_idxs2w


def load_tags(universal):
    if universal == 1:
        t_emb = np.load(os.path.abspath("NegNN/w2v/pos_50/pos_50_syn0.npy"))
        idxs2t_list = np.load(os.path.abspath("NegNN/w2v/pos_50/index2word.npy"))
        pre_t2idxs = dict([(_t, i) for i, _t in enumerate(idxs2t_list)])
        pre_idxs2t = dict([(v, k) for k, v in pre_t2idxs.items()])
    elif universal == 2:
        t_emb = np.load(os.path.abspath("NegNN/w2v/uni_50/uni_50_syn0.npy"))
        idxs2t_list = np.load(os.path.abspath("NegNN/w2v/uni_50/index2word.npy"))
        pre_t2idxs = dict([(_t, i) for i, _t in enumerate(idxs2t_list)])
        pre_idxs2t = dict([(v, k) for k, v in pre_t2idxs.items()])

    return t_emb, pre_t2idxs, pre_idxs2t


def pad_embeddings(w2v_we, emb_size):
    # add at <UNK> random vector at -2 and at -1 for padding
    w2v_we = np.vstack((w2v_we, 0.2 * np.random.uniform(-1.0, 1.0, [2, emb_size])))
    return w2v_we


def transform(tokens, _dict):
    return np.asarray([np.asarray([_dict[i] for i in r]) for r in tokens])


def get_index(w, _dict, voc_dim):
    return _dict[w] if w in _dict else voc_dim - 2


def load_train_dev(scope, event, lang, emb_dim, universal):
    # load data
    pre_w_emb, pre_w_voc, pre_w_voc_inv = load_words()
    # pad embedding with an <UNK> and a <PAD> vector
    pre_w_emb = pad_embeddings(pre_w_emb, emb_dim)

    # load training and dev data
    training = Data(fn_training)
    dev = Data(fn_dev)
    # get all strings
    sents, tags, tags_uni, labels, cues, scopes, lengths = data2sents(
        [training, dev], event, scope, lang
    )

    # create dictionaries for all except tags
    y2idxs = {"I": 0, "O": 1, "E": 2}

    words_idxs = [
        np.array(
            [get_index(w.lower(), pre_w_voc, pre_w_emb.shape[0]) for w in sent],
            dtype=np.int32,
        )
        for sent in sents
    ]
    y_idxs = [
        np.array([y2idxs[y] for y in y_array], dtype=np.int32) for y_array in labels
    ]
    cues_idxs = [
        np.array([1 if c == "CUE" else 0 for c in c_array], dtype=np.int32)
        for c_array in cues
    ]
    scope_idxs = [
        np.array([1 if s == "S" else 0 for s in s_array], dtype=np.int32)
        for s_array in scopes
    ]

    train_lex, valid_lex = words_idxs[: lengths[0]], words_idxs[lengths[0] :]
    train_cues, valid_cues = cues_idxs[: lengths[0]], cues_idxs[lengths[0] :]
    train_scope, valid_scope = scope_idxs[: lengths[0]], scope_idxs[lengths[0] :]
    train_y, valid_y = y_idxs[: lengths[0]], y_idxs[lengths[0] :]

    if universal in [1, 2]:
        pre_t_emb, pre_t_voc, pre_t_voc_inv = load_tags(universal)
        # pad embedding with an <UNK> and a <PAD> vector
        pre_t_emb = pad_embeddings(pre_t_emb, emb_dim)

        if universal == 1:
            tags_idxs = [
                np.array(
                    [get_index(t, pre_t_voc, pre_t_emb.shape[0]) for t in tag_sent],
                    dtype=np.int32,
                )
                for tag_sent in tags
            ]
        elif universal == 2:
            tags_idxs = [
                np.array(
                    [get_index(t, pre_t_voc, pre_t_emb.shape[0]) for t in tag_sent],
                    dtype=np.int32,
                )
                for tag_sent in tags_uni
            ]

        train_tags, valid_tags = tags_idxs[: lengths[0]], tags_idxs[lengths[0] :]

        train_set = train_lex, train_tags, train_tags, train_cues, train_scope, train_y
        valid_set = valid_lex, valid_tags, valid_tags, valid_cues, valid_scope, valid_y

        return train_set, valid_set, {"idxs2w": pre_w_voc_inv}, pre_w_emb, pre_t_emb

    else:
        train_set = train_lex, [], [], train_cues, train_scope, train_y
        valid_set = valid_lex, [], [], valid_cues, valid_scope, valid_y

        return train_set, valid_set, {"idxs2w": pre_w_voc_inv}, pre_w_emb, None


def load_test(fn_test, scope, event, lang, emb_dim, universal):
    # load data
    pre_w_emb, pre_w_voc, pre_w_voc_inv = load_words()
    # pad embedding with an <UNK> and a <PAD> vector
    pre_w_emb = pad_embeddings(pre_w_emb, emb_dim)

    test = reduce(lambda x, y: x + y, map(lambda z: Data(z), fn_test))
    sents, tags, tags_uni, labels, cues, scopes, _ = data2sents(
        [test], event, scope, lang
    )

    # create dictionaries for all except tags
    y2idxs = {"I": 0, "O": 1, "E": 2}

    test_lex = [
        np.array(
            [get_index(w.lower(), pre_w_voc, pre_w_emb.shape[0]) for w in sent],
            dtype=np.int32,
        )
        for sent in sents
    ]
    test_y = [
        np.array([y2idxs[y] for y in y_array], dtype=np.int32) for y_array in labels
    ]
    test_cues = [
        np.array([1 if c == "CUE" else 0 for c in c_array], dtype=np.int32)
        for c_array in cues
    ]
    test_scope = [
        np.array([1 if s == "S" else 0 for s in s_array], dtype=np.int32)
        for s_array in scopes
    ]

    if universal in [1, 2]:
        pre_t_emb, pre_t_voc, pre_t_voc_inv = load_tags(universal)
        # pad embedding with an <UNK> and a <PAD> vector
        pre_t_emb = pad_embeddings(pre_t_emb, emb_dim)

        if universal == 1:
            test_tags = [
                np.array(
                    [get_index(t, pre_t_voc, pre_t_emb.shape[0]) for t in tag_sent],
                    dtype=np.int32,
                )
                for tag_sent in tags
            ]
        elif universal == 2:
            test_tags = [
                np.array(
                    [get_index(t, pre_t_voc, pre_t_emb.shape[0]) for t in tag_sent],
                    dtype=np.int32,
                )
                for tag_sent in tags_uni
            ]

        test_set = test_lex, test_tags, test_tags, test_cues, test_scope, test_y

        return (
            test_set,
            {"idxs2w": pre_w_voc_inv, "idxs2t": pre_t_voc_inv},
            pre_w_emb,
            pre_t_emb,
        )
    else:
        test_set = test_lex, [], [], test_cues, test_scope, test_y

        return test_set, {"idxs2w": pre_w_voc_inv}, pre_w_emb, None
