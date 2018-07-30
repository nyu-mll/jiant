'''Code generating file of ELMo weights, where RNN weights are orthogonalized.'''
import torch
import numpy as np
import h5py

filename = 'elmo_2x4096_512_2048cnn_2xhighway_weights_ortho.hdf5' #Path of ELMo-weights file to be written.  This is expected to be a copy of the default AllenNLP ELMo weights file.  
f = h5py.File(filename, 'r+')

#Below is a list of h4py keys to all sets of weights in the LSTM.  This list is found using the `.visit()` method.
list_of_paths_to_LSTM_weights = ['Cell0/LSTMCell/W_0', 'Cell0/LSTMCell/W_P_0', 'Cell1/LSTMCell/W_0', 'Cell1/LSTMCell/W_P_0']

for i in range(2): #BI-LSTM, so two RNNs
    for path in list_of_paths_to_LSTM_weights:
        dataset = f['RNN_' + str(i) +  '/RNN/MultiRNNCell/' + path]
        shape = dataset.shape
        data_tensor = torch.tensor(np.array(dataset[...]))
        dataset[...] = (torch.nn.init.orthogonal_(data_tensor)).numpy()

f.close()
