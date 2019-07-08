"""
LSTMCell used in the Reading Network of PRPN
Reference: Parsing-Reading-Predict Networks (PRPN; Shen et al., 2018)
All the modules in this file are taken without change from: https://github.com/yikangshen/PRPN
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import *


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, dropout=0):
        super(LSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size, bias), LayerNorm(4 * hidden_size)
        )
        self.hh = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias), LayerNorm(4 * hidden_size)
        )
        self.c_norm = LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, input, hidden):

        hx, cx = hidden
        gates = self.ih(input) + self.hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * F.tanh(self.c_norm(cy))

        return hy, cy
