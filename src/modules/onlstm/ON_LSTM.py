import torch.nn.functional as F
import torch.nn as nn
import torch
from src.modules.onlstm.locked_dropout import LockedDropout

import numpy as np
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


def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
    padding_idx = -1
  #padding_idx=-1
  X = torch.nn.functional.embedding(words, masked_embed_weight,
    padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
  )
  return X

if __name__ == '__main__':
  V = 50
  h = 4
  bptt = 10
  batch_size = 2

  embed = torch.nn.Embedding(V, h)

  words = np.random.random_integers(low=0, high=V-1, size=(batch_size, bptt))
  words = torch.LongTensor(words)

  origX = embed(words)
  X = embedded_dropout(embed, words)

  print(origX)
  print(X)

class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)


def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_size, dropconnect=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2, bias=True),
            # LayerNorm(3 * hidden_size)
        )
        self.hh = LinearDropConnect(hidden_size, hidden_size*4+self.n_chunk*2, bias=True, dropout=dropconnect)

        # self.c_norm = LayerNorm(hidden_size)

        self.drop_weight_modules = [self.hh]

    def forward(self, input, hidden,
                transformed_input=None):
        hx, cx = hidden

        if transformed_input is None:
            transformed_input = self.ih(input)
        gates = transformed_input + self.hh(hx)
        cingate, cforgetgate = gates[:, :self.n_chunk*2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:,self.n_chunk*2:].view(-1, self.n_chunk*4, self.chunk_size).chunk(4,1)

        cingate = 1. - cumsoftmax(cingate)
        cforgetgate = cumsoftmax(cforgetgate)

        distance_cforget = 1. - cforgetgate.sum(dim=-1) / self.n_chunk
        distance_cin = cingate.sum(dim=-1) / self.n_chunk

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cell = F.tanh(cell)
        outgate = F.sigmoid(outgate)

        # cy = cforgetgate * forgetgate * cx + cingate * ingate * cell

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell

        # hy = outgate * F.tanh(self.c_norm(cy))
        hy = outgate * F.tanh(cy)
        return hy.view(-1, self.hidden_size), cy, (distance_cforget, distance_cin)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.n_chunk, self.chunk_size).zero_())

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


class ONLSTMStack(nn.Module):
    def __init__(self, layer_sizes, chunk_size, dropout=0., dropconnect=0., embedder=None, phrase_layer=None, dropouti=0.5, dropoutw=0.1, dropouth=0.3, batch_size=20):
        super(ONLSTMStack, self).__init__()
        self.layer_sizes=layer_sizes
        self.cells = nn.ModuleList([ONLSTMCell(layer_sizes[i],
                                               layer_sizes[i+1],
                                               chunk_size,
                                               dropconnect=dropconnect)
                                    for i in range(len(layer_sizes) - 1)])
        self.lockdrop = LockedDropout()
        #self.lockdrop2= LockedDropout()
        self.dropout = dropout
        self.dropouti=dropouti
        self.dropouth=dropouth
        self.sizes = layer_sizes
        self.embedder=embedder
        dim=self.embedder.token_embedder_words.weight.shape
        self.emb=nn.Embedding(dim[0], dim[1])
        #self.hidden=self.init_hidden(batch_size)
        self._phrase_layer=phrase_layer

        self.dropoutw=dropoutw

    def get_input_dim(self):
        return self.layer_sizes[0]

    def get_output_dim(self):
        return self.layer_sizes[-1]

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells]

    def forward(self, input, task=None):
        #input= torch.transpose(input, 0, 1)
        batch_size=input.size()[1]
        #hidden=self.hidden
        hidden = self.init_hidden(batch_size)
        return self.forward_actual(input, hidden)

    def forward_actual(self, input, hidden):
        abs_inp=input
        input=embedded_dropout(self.emb, input, dropout=self.dropoutw if self.training else 0)
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
        output=self.lockdrop(output, self.dropout)
        return output, torch.ones(output.shape)
        #import pdb;pdb.set_trace()
        #return output, prev_state, raw_outputs, outputs, (torch.stack(distances_forget), torch.stack(distances_in))


if __name__ == "__main__":
    x = torch.Tensor(10, 10, 10)
    x.data.normal_()
    lstm = LSTMCellStack([10, 10, 10])
    print(lstm(x, lstm.init_hidden(10))[1])

