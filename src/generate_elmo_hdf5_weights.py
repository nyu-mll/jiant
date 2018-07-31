'''Code generating file of ELMo weights, where RNN weights are either randomized or orthogonalized.'''
import torch
import numpy as np
import h5py

import h5py_utils

elmo_model = 'ortho' #Type of ELMo RNN weights: 'random' or 'ortho'.
default_ELMo_weight_filepath ='/nfs/jsalt/share/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
ELMo_weight_filename = 'elmo_2x4096_512_2048cnn_2xhighway_weights_' + elmo_model + '.hdf5' #Name of ELMo-weights file to be written.

ELMo_weight_file = h5py_utils.copy_h5py_file(default_ELMo_weight_filepath, ELMo_weight_filename)

#Below is a list of h4py keys to all sets of weights in the LSTM.  This list is found using the `.visit()` method.
list_of_paths_to_LSTM_weights = ['Cell0/LSTMCell/B', 'Cell0/LSTMCell/W_0', 'Cell0/LSTMCell/W_P_0', 'Cell1/LSTMCell/B', 'Cell1/LSTMCell/W_0', 'Cell1/LSTMCell/W_P_0']

for i in range(2): #BI-LSTM, so two RNNs
    for path in list_of_paths_to_LSTM_weights:
        weight_tensor = ELMo_weight_file['RNN_' + str(i) +  '/RNN/MultiRNNCell/' + path]
        if elmo_model == 'random' or path[-1] == 'B': #In case when we want to randomize RNN weights (bias-term weights are randomized in both 'random' and 'ortho').
            shape = weight_tensor.shape
            weight_tensor[...] = np.random.normal(0, 1, list(shape))
        elif elmo_model == 'ortho': #In case when we want to convert RNN weights to orthogonal matrices.
            weight_torch_tensor = torch.tensor(np.array(weight_tensor[...]))
            weight_tensor[...] = (torch.nn.init.orthogonal_(weight_torch_tensor)).numpy()
        else:
            raise RuntimeError("Failed to recognize flag elmo_model = " + elmo_model)
ELMo_weight_file.close()
