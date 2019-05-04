#!/usr/bin/env python

# Helper script to generate ELMo weights file, where RNN weights are either randomized or
#   randomized+orthogonalized..
# Uses h5py to read in default ELMo weights file, make copy of it, and modify and save the copy.
#
# Usage:
#  python generate_elmo_hdf5_weights.py -m random -s 0 -o name_of_weights_file.hdf5
#
# Speed: takes around 6 seconds to generate random ELMo weight file and 10
# seconds to generate orthogonal ELMo weight file.

import argparse
import sys

import h5py
import numpy as np
import torch

import h5py_utils

ELMO_WEIGHTS_PATH = "/nfs/jsalt/share/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        dest="elmo_model",
        type=str,
        required=True,
        help="Type of ELMo weights: random or ortho.",
    )
    parser.add_argument(
        "-s",
        dest="seed",
        type=int,
        required=True,
        default=0,
        help="Random seed for modifying RNN weights.",
    )
    parser.add_argument(
        "-o",
        dest="output_filename",
        type=str,
        required=True,
        help="Name of the output (hdf5 file containing modified ELMo weights).",
    )
    args = parser.parse_args(args)

    elmo_model = args.elmo_model  # Type of ELMo RNN weights: 'random' or 'ortho'.
    np.random.seed(args.seed)
    ELMo_weight_filename = args.output_filename  # ELMo weights file to be written.

    assert elmo_model in ["random", "ortho"], "Failed to recognize flag elmo_model = " + elmo_model

    # ELMo_weight_filename = 'elmo_2x4096_512_2048cnn_2xhighway_weights_' +
    # elmo_model + '_seed_' + str(args.seed) + '.hdf5' #Name of ELMo-weights
    # file to be written.

    ELMo_weight_file = h5py_utils.copy_h5py_file(ELMO_WEIGHTS_PATH, ELMo_weight_filename)

    # Below is a list of h4py keys to all sets of weights in the LSTM.  This
    # list is found using the `.visit()` method.
    list_of_paths_to_LSTM_weights = [
        "Cell0/LSTMCell/B",
        "Cell0/LSTMCell/W_0",
        "Cell0/LSTMCell/W_P_0",
        "Cell1/LSTMCell/B",
        "Cell1/LSTMCell/W_0",
        "Cell1/LSTMCell/W_P_0",
    ]

    for i in range(2):  # BI-LSTM, so two RNNs
        for path in list_of_paths_to_LSTM_weights:
            weight_tensor = ELMo_weight_file["RNN_" + str(i) + "/RNN/MultiRNNCell/" + path]
            shape = weight_tensor.shape
            weight_tensor[...] = np.random.normal(0, 1, list(shape))
            # In case when we want to convert RNN weights to orthogonal matrices:
            if elmo_model == "ortho" and (not path.endswith("/B")):
                weight_torch_tensor = torch.tensor(np.array(weight_tensor[...]))
                weight_tensor[...] = (torch.nn.init.orthogonal_(weight_torch_tensor)).numpy()
    ELMo_weight_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
