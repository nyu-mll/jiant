from itertools import islice

import random
import tensorflow as tf
import pickle
import numpy as np
import os


def load(fname):
    """fname:: file name of the pickled dict where data is stored
    returns a tuple containing training, dev, test sets and word2idx
    dictionaries"""

    with open(fname, "rb") as f:
        train_set, valid_set, test_set, dicts = pickle.load(f)
    return train_set, valid_set, test_set, dicts


def shuffle(lol, seed):
    """
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    """
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def minibatch(l, bs):
    """
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    """
    out = [l[:i] for i in xrange(1, min(bs, len(l) + 1))]
    out += [l[i - bs : i] for i in xrange(bs, len(l) + 1)]
    assert len(l) == len(out)
    return out


def contextwin(l, win, flag, idx=None):
    """
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    """
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    if flag == "cue":
        lpadded = win / 2 * [0] + l + win / 2 * [0]
    else:
        lpadded = win / 2 * [idx] + l + win / 2 * [idx]
    out = [lpadded[i : i + win] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def contextwin_lr(l, win_left, win_right, flag, idx=None):
    """
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    """

    l = list(l)

    if flag == "cue":
        lpadded = win_left * [0] + l + win_right * [0]
    else:
        lpadded = win_left * [idx] + l + win_right * [idx]
    # print lpadded
    out = [lpadded[i : i + win_right + win_left + 1] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def padding(l, max_len, pad_idx, x=True):
    if x:
        pad = [pad_idx] * (max_len - len(l))
    else:
        pad = [[0, 1]] * (max_len - len(l))
    return np.concatenate((l, pad), axis=0)


def random_uniform(shape, name, low=-1.0, high=1.0, update=True):
    return tf.Variable(
        0.2 * tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32),
        name=name,
        trainable=update,
    )


def unpickle_data(folder):
    with open(os.path.join(folder, "train_dev.pkl"), "wb") as f:
        return pickle.load(f)
