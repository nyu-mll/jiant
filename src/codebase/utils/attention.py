from collections import namedtuple

import math
import torch
from gtd.ml.torch.utils import GPUVariable
from torch.nn import Parameter
from torch.nn import Softmax, Tanh, Module

from gtd.ml.torch.utils import conditional, NamedTupleLike

# TODO at some point will likely want to pass weights

class AttentionOutput(namedtuple('AttentionOutput', ['weights', 'context']), NamedTupleLike):
    pass
"""
Attributes:
    weights (Variable): of shape (batch_size, num_cells)
    context (Variable): of shape (batch_size, memory_dim)
"""

class DummyAttention(Module):
    def __init__(self, memory_dim, query_dim, attn_dim):
        super(DummyAttention, self).__init__()
        self.memory_dim = memory_dim
        self.query_dim = query_dim
        self.attn_dim = attn_dim

    def forward(self, memory_cells, query):
        batch_size, num_cells = memory_cells.mask.size()
        weights = GPUVariable(torch.zeros(batch_size, num_cells))
        context = GPUVariable(torch.zeros(batch_size, self.memory_dim))
        return context #AttentionOutput(weights=weights, context=context)



class Attention(Module):
    def __init__(self, memory_dim, query_dim, attn_dim):
        super(Attention, self).__init__()
        self.tanh = Tanh()
        self.softmax = Softmax()

        self.memory_dim = memory_dim
        self.query_dim = query_dim
        self.attn_dim = attn_dim

        self.memory_transform = Parameter(self._initialize_weight_matrix(memory_dim, attn_dim))  # Wh
        self.query_transform = Parameter(self._initialize_weight_matrix(query_dim, attn_dim))  # Ws
        self.v_transform = Parameter(self._initialize_weight_matrix(attn_dim, 1))  # v

    @classmethod
    def _initialize_weight_matrix(cls, in_dim, out_dim):
        stdv = 1. / math.sqrt(in_dim)
        m = torch.ones(in_dim, out_dim)
        m.uniform_(-stdv, stdv)
        return m

    def forward(self, memory_cells, query):
        """Generates a density over a set of elements w.r.t. the query vector.

        Et(i) = tanh(Hi * Wh + St * Ws) * v
        At = softmax(Et)

        Dimensions:
            Hi: (batch_size x memory_dim)
            St: (batch_size x query_dim)
            Wh: (memory_dim x attn_dim)
            Ws: (query_dim x attn_dim)
            v:  (attn_dim x 1)
            --
            tanh( Hi * Wh + St * Ws ):       (batch_size x attn_dim)
            tanh( Hi * Wh + St * Ws ) * v:   (batch_size x 1)
            At = softmax(Et):                (batch_size x num_cells)

        Args:
            memory_cells (SequenceBatch): (batch_size x num_cells x memory_dim)
            query (torch.Variable): St (batch_size x query_dim)

        Returns:
            Variable: (batch_size x num_cells) array
        """
        transformed_query = torch.mm(query, self.query_transform)  # (batch_size, attn_dim)

        batch_size, num_cells = memory_cells.mask.size()
        memory_cells_ = torch.transpose(memory_cells.values, 0, 1)  # (num_cells, batch_size, memory_dim)
        expanded_transformed_query = transformed_query.expand(num_cells, batch_size, self.attn_dim)
        expanded_memory_transform = self.memory_transform.expand(num_cells, self.memory_dim, self.attn_dim)
        expanded_v_transform = self.v_transform.expand(num_cells, self.attn_dim, 1)

        # (num_cells, batch_size, attn_dim)
        attn_embeds = torch.bmm(memory_cells_, expanded_memory_transform) + expanded_transformed_query
        attn_embeds = self.tanh(attn_embeds)
        attn_embeds = torch.bmm(attn_embeds, expanded_v_transform)  # (num_cells, batch_size, 1)
        logits = torch.transpose(attn_embeds.squeeze(2), 0, 1)

        mask = memory_cells.mask

        # no_cells is a FloatTensor with shape (batch_size, num_cells)
        # no_cells[i, j] = 1 if example i has NO memory cells, 0 otherwise
        no_cells = (1 - mask).prod(1).expand_as(mask)
        # TODO(kelvin): check for numerical stability. Product of 1's does not necessarily equal 1 exactly, which we need

        suppress = GPUVariable(torch.zeros(*mask.size()))
        suppress[mask == 0] = float('-inf')  # send the logit of non-cells to -infinity
        suppress[no_cells == 1] = 0.0  # but if an entire row has no cells, just leave the cells alone

        logits = logits + suppress
        # -inf + anything = -inf

        # compute normalized weights
        weights = self.softmax(logits)  # (batch_size, num_cells)

        # if a given row has no memory cells, weights should be all zeros
        all_zeros = GPUVariable(torch.zeros(*mask.size()))
        weights = conditional(no_cells, all_zeros, weights)

        context = torch.bmm(weights.unsqueeze(1), memory_cells.values)  # (batch_size, 1, memory_dim)
        context = context.squeeze(1)  # (batch_size, memory_dim)
        return context #AttentionOutput(weights=weights, context=context)
