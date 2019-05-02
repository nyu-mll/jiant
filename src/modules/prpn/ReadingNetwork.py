"""
Reading Network (LSTMN with self-attention) of PRPN
Reference: Parsing-Reading-Predict Networks (PRPN; Shen et al., 2018)
All the modules in this file are taken without change from: https://github.com/yikangshen/PRPN
"""
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from .blocks import softmax
from .LSTMCell import LSTMCell


class ReadingNetwork(nn.Module):
    def __init__(self, ninp, nout, nslots, dropout, idropout):
        super(ReadingNetwork, self).__init__()

        self.ninp = ninp
        self.nout = nout
        self.nslots = nslots
        self.drop = nn.Dropout(dropout)
        self.memory_rnn = LSTMCell(ninp, nout, bias=True, dropout=0)
        self.projector_summ = nn.Sequential(
            nn.Dropout(idropout), nn.Linear(ninp + nout, nout), nn.Dropout(idropout)
        )

    def forward(self, input, memory, gate_time, rmask):
        memory_h, memory_c = memory

        # attention
        selected_memory_h, selected_memory_c, attention0 = self.attention(
            input, memory_h, memory_c, gate=gate_time
        )

        # recurrent
        input = self.drop(input)
        h_i, c_i = self.memory_rnn(input, (selected_memory_h * rmask, selected_memory_c))

        # updata memory
        memory_h = torch.cat([h_i[:, None, :], memory_h[:, :-1, :]], dim=1)
        memory_c = torch.cat([c_i[:, None, :], memory_c[:, :-1, :]], dim=1)

        return h_i, (memory_h, memory_c), attention0

    def attention(self, input, memory_h, memory_c, gate=None):
        # select memory to use
        key = self.projector_summ(torch.cat([input, memory_h[:, 0, :]], dim=1))
        logits = torch.bmm(memory_h, key[:, :, None]).squeeze(2)
        logits = logits / math.sqrt(self.nout)
        attention = softmax(logits, gate)
        selected_memory_h = (memory_h * attention[:, :, None]).sum(dim=1)
        selected_memory_c = (memory_c * attention[:, :, None]).sum(dim=1)
        return selected_memory_h, selected_memory_c, attention

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(bsz, self.nslots, self.nout).zero_()),
            Variable(weight.new(bsz, self.nslots, self.nout).zero_()),
        )
