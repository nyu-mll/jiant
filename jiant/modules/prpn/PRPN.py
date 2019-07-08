"""
Code for Parsing-Reading-Predict Networks (PRPN; Shen et al., 2018)
This file is a version of a class from https://github.com/yikangshen/PRPN, modified to integrate with jiant.
We modified the forward function of original PRPN code.
"""
import torch
import torch.nn as nn

from .ParsingNetwork import ParsingNetwork
from .PredictNetwork import PredictNetwork
from .ReadingNetwork import ReadingNetwork


class PRPN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        ninp,
        nhid,
        nlayers,
        nslots=15,
        nlookback=5,
        resolution=0.1,
        embedder=None,
        dropout=0.7,
        idropout=0.5,
        rdropout=0.5,
        phrase_layer=None,
        tie_weights=True,
        hard=True,
        res=0,
        batch_size=20,
    ):
        super(PRPN, self).__init__()

        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers
        self.nslots = nslots
        self.nlookback = nlookback

        self.drop = nn.Dropout(dropout)
        self.idrop = nn.Dropout(idropout)
        self.rdrop = nn.Dropout(rdropout)

        # Feedforward layers

        self.embedder = embedder
        dim = self.embedder.token_embedder_words.weight.shape

        self._phrase_layer = phrase_layer

        self.ninp = ninp
        self.emb = nn.Embedding(dim[0], self.ninp)
        self.parser = ParsingNetwork(self.ninp, nhid, nslots, nlookback, resolution, idropout, hard)
        self.reader = nn.ModuleList(
            [ReadingNetwork(self.ninp, nhid, nslots, dropout=dropout, idropout=idropout)]
            + [
                ReadingNetwork(nhid, nhid, nslots, dropout=idropout, idropout=idropout)
                for i in range(nlayers - 1)
            ]
        )
        self.predictor = PredictNetwork(nhid, self.ninp, nslots, idropout, res)
        # self.decoder = nn.Linear(ninp, ntoken)

        # if tie_weights:
        #    self.decoder.weight = self.encoder.weight

        self.attentions = None
        self.gates = None

        self.init_weights()

    def get_input_dim(self):
        return self.ninp

    def get_output_dim(self):
        return self.ninp

    def init_weights(self):
        initrange = 0.01
        self.emb.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def clip_grad_norm(self, clip):
        for model in self.reader:
            torch.nn.utils.clip_grad_norm(model.memory_rnn.parameters(), clip)

    def forward(self, input, task=None):
        batch_size = input.size()[1]
        hidden = self.init_hidden(batch_size)
        return self.forward_actual(input, hidden)

    def forward_actual(self, input, hidden_states):
        abs_inp = input
        ntimestep = input.size(0)
        bsz = input.size(1)
        emb = self.emb(input)  # timesteps, bsz, ninp
        output_h = []
        output_memory = []
        attentions = []

        reader_state, parser_state, predictor_state = hidden_states  # memory_h: bsz, nslots, nhid

        (memory_gate, memory_gate_next), gate, parser_state = self.parser(emb, parser_state)

        rmask = torch.autograd.Variable(torch.ones(self.nlayers, self.nhid))
        if input.is_cuda:
            rmask = rmask.cuda()
        rmask = self.rdrop(rmask)

        for i in range(input.size(0)):
            emb_i = emb[i]  # emb_i: bsz, nhid
            attention = []
            attention.append(memory_gate[i])

            # summarize layer
            h_i = emb_i
            for j in range(self.nlayers):
                hidden = reader_state[j]

                h_i, new_memory, attention0 = self.reader[j](h_i, hidden, memory_gate[i], rmask[j])

                # updata states
                attention.append(attention0)
                reader_state[j] = new_memory

            # predict layer
            selected_memory_h, predictor_state, attention1 = self.predictor.attention(
                h_i, predictor_state, gate_time=memory_gate_next[i]
            )
            output_h.append(h_i)
            output_memory.append(selected_memory_h)

            attention.append(memory_gate_next[i])
            attention.append(attention1)
            attentions.append(torch.stack(attention, dim=1))

        self.attentions = torch.stack(attentions, dim=0)
        self.gates = gate

        output_h = torch.stack(output_h, dim=0)
        output_memory = torch.stack(output_memory, dim=0)
        output = self.predictor(output_h.view(-1, self.nhid), output_memory.view(-1, self.nhid))

        output = self.drop(output).view(ntimestep, bsz, -1)
        # decoded = self.decoder(output)
        # return decoded.view(ntimestep, bsz, -1), (reader_state, parser_state, predictor_state)
        mask = abs_inp != 0

        return output, mask

    def init_hidden(self, bsz):
        return (
            [self.reader[i].init_hidden(bsz) for i in range(self.nlayers)],
            self.parser.init_hidden(bsz),
            self.predictor.init_hidden(bsz),
        )
