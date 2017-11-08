from abc import ABCMeta, abstractmethod, abstractproperty
import pdb
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F

from codebase.utils.utils import tile_state, gated_update
from codebase.utils.seq_batch import SequenceBatchElement

# TODO handle masks somehow

class Encoder(nn.Module):
    '''
    Abstract class for a sentence encoder
    '''
    __metaclass__ = ABCMeta

    @abstractproperty
    def hid_dim(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        '''
        Embed a source sequence

        Ins:
            inputs (list[TODO]): where each element is of shape
                                 (batch_size, input_dim)
        Outs:
            hid_states (list[TODO]): where each element is 
                                     (batch_size, hid_dim)
        '''
        raise NotImplementedError



class RNNEncoder(Encoder):
    '''
    Prototype encoder; the first half of the decoder in the CVAE. 
        Implemented as an LSTM (half seq2seq w/ attention).
    '''

    def __init__(self, rnn_cell):
        '''
        Ins:
            rnn_cell ()
        '''
        super(RNNEncoder, self).__init__()

        self.rnn_cell = rnn_cell
        self._hid_dim = hid_dim = rnn_cell.hidden_size
        self.h0 = Parameter(torch.zeros(hid_dim))
        self.c0 = Parameter(torch.zeros(hid_dim))
        self._hid_dim = hid_dim

    @property
    def hid_dim(self):
        return self._hid_dim

    def forward(self, inputs):
        '''
        Ins:
            inputs: (list[SequenceBatchElement]): each element has shape
                                                  (batch_size, input_dim)
            TODO(Alex): masks???

        Outs:
            hid_states (list[SequenceBatchElement]): each element has shape
                                                     (batch_size, hid_dim)
        '''

        batch_size = inputs[0].values.size()[0]
        h = tile_state(self.h0, batch_size)
        c = tile_state(self.c0, batch_size)
        hid_states = []
        for t, x in enumerate(inputs):
            h_t, c_t = self.rnn_cell(x.values, (h, c))
            h = gated_update(h, h_t, x.mask)
            c = gated_update(c, c_t, x.mask)
            hid_states.append(SequenceBatchElement(h, x.mask))
        return hid_states



class BidirectionalRNNEncoder(Encoder):
    '''
    Bidirectional RNN
    '''

    def __init__(self, input_dim, hid_dim, rnn_cell_factory):
        super(BidirectionalRNNEncoder, self).__init__()

        if hid_dim % 2 != 0:
            raise ValueError('hid_dim must be even for BidirectionalRNNEncoder.')
        self._hid_dim = hid_dim

        build_encoder = lambda: RNNEncoder(rnn_cell_factory(
                           input_dim, int(hid_dim / 2)))
        self.forward_encoder = build_encoder()
        self.backward_encoder = build_encoder()

    @property
    def hid_dim(self):
        return self._hid_dim

    def forward(self, inputs):
        '''
        Compute bidirectional RNN embeddings

        In:
            inputs (list[SequenceBatchElement])

        Outs:
            forward_states (list[SequenceBatchElement])
            backward_states (list[SequenceBatchElement])
        
        '''
        reverse = lambda seq: list(reversed(seq))
        forward_states = self.forward_encoder(inputs)
        backward_states = reverse(self.backward_encoder(reverse(inputs)))
        return forward_states, backward_states



class MultiLayerRNNEncoder(Encoder):
    '''
    Multilayer RNN
    '''

    def __init__(self, input_dim, hid_dim, n_layers, rnn_cell_factory):
        '''
        Args:
            input_dim (int)
            hid_dim (int)
            n_layers (int)
            rnn_cell_factory (Callable[[int, int], RNNCell)
        '''
        super(MultiLayerRNNEncoder, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for layer in range(n_layers):
            in_dim = input_dim if layer == 0 else hid_dim
            out_dim = hid_dim
            encoder = BidirectionalRNNEncoder(in_dim, out_dim, 
                                              rnn_cell_factory)
            self.add_module('encoder_layer_{}'.format(layer), encoder)
            self.layers.append(encoder)

    @property
    def hid_dim(self):
        return self.layers[-1].hid_dim


    def forward(self, inputs):
        '''
        Ins:
            inputs: (list[Variable]): each element has shape
                                      (batch_size, input_dim)
            TODO(Alex): masks???

        Outs:
            hid_states (list[Variable]): each element has shape
                                         (batch_size, hid_dim)
        '''
        for i, layer in enumerate(self.layers):
            if i == 0:
                prev_hid_states = inputs
            else:
                prev_hid_states = [torch.cat([f, b], 1) for f, b in 
                                   zip(forward_states, backward_states)]

            new_forward_states, new_backward_states = \
                        layer(prev_hid_states)

            if i == 0:
                # no skip connections because dimensions don't match
                forward_states, backward_states = \
                        new_forward_states, new_backward_states
            else:
                # add residuals
                add_residuals = lambda a_list, b_list: [a + b for \
                                            a, b in zip(a_list, b_list)]

                forward_states = add_residuals(forward_states, new_forward_states)
                backward_states = add_residuals(backward_states, new_backward_states)

        return forward_states, backward_states
