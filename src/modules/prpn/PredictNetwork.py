"""
Prediction Network of PRPN, that predicts the language model probabilities
Reference: Parsing-Reading-Predict Networks (PRPN; Shen et al., 2018)
All the modules in this file are taken without change from: https://github.com/yikangshen/PRPN
"""
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from .blocks import ResBlock, softmax


class PredictNetwork(nn.Module):
    def __init__(self, ninp, nout, nslots, dropout, nlayers=1):
        super(PredictNetwork, self).__init__()

        self.ninp = ninp
        self.nout = nout
        self.nslots = nslots
        self.nlayers = nlayers

        self.drop = nn.Dropout(dropout)

        self.projector_pred = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(ninp, ninp), nn.Dropout(dropout)
        )

        if nlayers > 0:
            self.res = ResBlock(ninp * 2, nout, dropout, nlayers)
        else:
            self.res = None

        self.ffd = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(ninp * 2, nout), nn.BatchNorm1d(nout), nn.Tanh()
        )

    def forward(self, input, input_memory):
        input = torch.cat([input, input_memory], dim=1)
        if self.nlayers > 0:
            input = self.res(input)
        output = self.ffd(input)
        return output

    def attention(self, input, memory, gate_time):
        key = self.projector_pred(input)
        # select memory to use
        logits = torch.bmm(memory, key[:, :, None]).squeeze(2)
        logits = logits / math.sqrt(self.ninp)
        attention = softmax(logits, gate_time)
        selected_memory_h = (memory * attention[:, :, None]).sum(dim=1)
        memory = torch.cat([input[:, None, :], memory[:, :-1, :]], dim=1)
        return selected_memory_h, memory, attention

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        self.ones = Variable(weight.new(bsz, 1).zero_() + 1.0)
        return Variable(weight.new(bsz, self.nslots, self.ninp).zero_())
