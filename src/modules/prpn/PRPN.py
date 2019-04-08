import torch
import torch.nn as nn

from ParsingNetwork import ParsingNetwork
from PredictNetwork import PredictNetwork
from ReadingNetwork import ReadingNetwork


class PRPN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers,
                 nslots=5, nlookback=1, resolution=0.1,
                 dropout=0.4, idropout=0.4, rdropout=0.1,
                 tie_weights=False, hard=False, res=1):
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
        self.encoder = nn.Embedding(ntoken, ninp)
        self.parser = ParsingNetwork(ninp, nhid, nslots, nlookback, resolution, idropout, hard)
        self.reader = nn.ModuleList([ReadingNetwork(ninp, nhid, nslots, dropout=dropout, idropout=idropout), ] +
                                    [ReadingNetwork(nhid, nhid, nslots, dropout=idropout, idropout=idropout)
                                     for i in range(nlayers - 1)])
        self.predictor = PredictNetwork(nhid, ninp, nslots, idropout, res)
        self.decoder = nn.Linear(ninp, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.attentions = None
        self.gates = None

        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


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
        emb = self.encoder(input)  # timesteps, bsz, ninp
        output_h = []
        output_memory = []
        attentions = []

        reader_state, parser_state, predictor_state = hidden_states  # memory_h: bsz, nslots, nhid

        (memory_gate, memory_gate_next), gate, parser_state = self.parser(emb, parser_state)

        rmask = torch.autograd.Variable(torch.ones(self.nlayers, self.nhid))
        if input.is_cuda: rmask = rmask.cuda()
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
            selected_memory_h, predictor_state, attention1 = self.predictor.attention(h_i, predictor_state,
                                                                                      gate_time=memory_gate_next[i])
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

        output = self.drop(output)
        decoded = self.decoder(output)
        #return decoded.view(ntimestep, bsz, -1), (reader_state, parser_state, predictor_state)
        mask = abs_inp != 0

        return output, mask

    def init_hidden(self, bsz):
        return [self.reader[i].init_hidden(bsz)
                for i in range(self.nlayers)], \
               self.parser.init_hidden(bsz), \
               self.predictor.init_hidden(bsz)



class ONLSTMStack(nn.Module):
    """
    ON-LSTM encoder composed of multiple ON-LSTM layers. Each layer is constructed
    through ONLSTMCell structures.
    Code credits: https://github.com/yikangshen/Ordered-Neurons
    """
    def __init__(self, layer_sizes, chunk_size, dropout=0., dropconnect=0., embedder=None, phrase_layer=None, dropouti=0.5, dropoutw=0.1, dropouth=0.3, batch_size=20):
        super(ONLSTMStack, self).__init__()
        self.layer_sizes = layer_sizes
        self.cells = nn.ModuleList([ONLSTMCell(layer_sizes[i],
                                               layer_sizes[i+1],
                                               chunk_size,
                                               dropconnect=dropconnect)
                                    for i in range(len(layer_sizes) - 1)])
        self.lockdrop = LockedDropout()
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.sizes = layer_sizes
        self.embedder = embedder
        dim = self.embedder.token_embedder_words.weight.shape
        self.emb = nn.Embedding(dim[0], dim[1])
        self._phrase_layer = phrase_layer

        self.dropoutw = dropoutw

    def get_input_dim(self):
        return self.layer_sizes[0]

    def get_output_dim(self):
        return self.layer_sizes[-1]

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells]

    def forward(self, input, task=None):
        batch_size = input.size()[1]
        hidden = self.init_hidden(batch_size)
        return self.forward_actual(input, hidden)

    def forward_actual(self, input, hidden):
        abs_inp = input
        input = embedded_dropout(self.emb, input, dropout=self.dropoutw if self.training else 0)
        input = self.lockdrop(input, self.dropouti)
        length, batch_size, _ = input.size()
        if self.training:
            for c in self.cells:
                c.sample_masks()

        prev_state = list(hidden)
        prev_layer = input

        raw_outputs = []
        outputs = []
        distances_forget = []
        distances_in = []
        for l in range(len(self.cells)):
            curr_layer = [None] * length
            dist = [None] * length
            t_input = self.cells[l].ih(prev_layer)
            for t in range(length):
                hidden, cell, d = self.cells[l](
                    None, prev_state[l],
                    transformed_input=t_input[t]
                )
                prev_state[l] = hidden, cell  # overwritten every timestep
                curr_layer[t] = hidden
                dist[t] = d

            prev_layer = torch.stack(curr_layer)
            dist_cforget, dist_cin = zip(*dist)
            dist_layer_cforget = torch.stack(dist_cforget)
            dist_layer_cin = torch.stack(dist_cin)
            raw_outputs.append(prev_layer)
            if l < len(self.cells) - 1:
                prev_layer = self.lockdrop(prev_layer, self.dropouth)
            outputs.append(prev_layer)
            distances_forget.append(dist_layer_cforget)
            distances_in.append(dist_layer_cin)
        output = prev_layer
        output = self.lockdrop(output, self.dropout)
        mask = abs_inp != 0
        self.distances = torch.stack(distances_forget)
        return output, mask
