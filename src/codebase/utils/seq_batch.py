from collections import namedtuple

import numpy as np
import torch
from torch.autograd import Variable

from codebase.utils.utils import GPUVariable, conditional, is_binary
from codebase.utils.utils import expand_dims_for_broadcast, NamedTupleLike


class SequenceBatch(namedtuple('SequenceBatch', ['values', 'mask']), NamedTupleLike):
    """
    Attributes:
        values (Variable): of shape (batch_size, max_seq_length, X1, X2, ...)
        mask (Variable[FloatTensor]): of shape (batch_size, max_seq_length)
    """
    __slots__ = ()
    def __new__(cls, values, mask):
        if not isinstance(values, Variable) or not isinstance(mask, Variable):
            raise ValueError('values and mask must both be of type Variable.')

        m = mask.data

        if len(m.size()) == 0:
            raise ValueError('Mask must not be 0-dimensional')

        # check that mask is binary
        if not is_binary(m):
            raise ValueError('Mask must be binary:\n{}'.format(mask))

        # check that mask is left-justified
        # since mask is binary, we just need to check that it is monotonically non-increasing from left to right
        batch_size, seq_len = m.size()
        if seq_len > 1:
            diffs = m[:, 1:] - m[:, :-1]  # (batch_size, max_seq_length - 1)
            non_increasing = diffs <= 0
            all_non_increasing = (torch.prod(non_increasing) == 1)
            if not all_non_increasing:
                raise ValueError('Mask must be left-justified:\n{}'.format(mask))

        self = super(SequenceBatch, cls).__new__(cls, values, mask)
        return self

    @classmethod
    def from_sequences(cls, sequences, tok2idx, min_seq_length=0):
        """Convert a batch of sequences into a SequenceBatch.

        Args:
            sequences (list[list[unicode]])
            tok2idx (dict)
            min_seq_length (int): enforce that the Tensor representing the SequenceBatch have at least
                this many columns.

        Returns:
            SequenceBatch
        """
        batch_size = len(sequences)
        if batch_size == 0:
            seq_length = 0
        else:
            seq_length = max(len(seq) for seq in sequences)  # max seq length in batch
        seq_length = max(seq_length, min_seq_length)  # make sure it is at least min_seq_length

        shape = (batch_size, seq_length)
        values = np.zeros(shape, dtype=np.int64)  # pad with zeros
        mask = np.zeros(shape, dtype=np.float32)
        for i, seq in enumerate(sequences):
            for j, word in enumerate(seq):
                values[i, j] = tok2idx[word] if word in tok2idx else tok2idx['<unk>']
                mask[i, j] = 1.0

        return SequenceBatch(GPUVariable(torch.from_numpy(values)), GPUVariable(torch.from_numpy(mask)))

    def split(self):
        """Convert SequenceBatch into a list of Variables, where each element represents one time step.

        Returns:
            list[SequenceBatchElement]: a list of SequenceBatchElements, where for each list element:
                element.values has shape (batch_size, X1, X2, ...)
                element.mask has shape (batch_size, 1)
        """
        values_list = [v.squeeze(dim=1) for v in self.values.split(1, dim=1)]
        mask_list = self.mask.split(1, dim=1)
        return [SequenceBatchElement(v, m) for v, m in zip(values_list, mask_list)]

    @classmethod
    def cat(cls, elements):
        """Concatenate SequenceBatchElements to form a SequenceBatch.

        Args:
            elements (list[SequenceBatchElement])

        Returns:
            SequenceBatch
        """
        values = torch.cat([e.values.unsqueeze(1) for e in elements], 1)
        mask = torch.cat([e.mask for e in elements], 1)
        return SequenceBatch(values, mask)

    @classmethod
    def weighted_sum(cls, seq_batch, weights):
        """Compute weighted sum of elements in a SequenceBatch.

        Args:
            seq_batch (SequenceBatch): with values of shape (batch_size, seq_length, X1, X2, ...)
            weights (Variable): of shape (batch_size, seq_length)

        Returns:
            Variable: of shape (batch_size, X1, X2, ...)
        """
        values = seq_batch.values
        mask = seq_batch.mask
        weights = weights * mask  # ignore weights outside mask
        weights = expand_dims_for_broadcast(weights, values).expand(values.size())
        weighted = values * weights
        return torch.sum(weighted, dim=1).squeeze(dim=1)

    @classmethod
    def reduce_sum(cls, seq_batch):
        weights = GPUVariable(torch.ones(*seq_batch.mask.size()))
        return cls.weighted_sum(seq_batch, weights)

    @classmethod
    def reduce_prod(cls, seq_batch):
        """Compute the product of each sequence in a SequenceBatch.
        
        If a sequence is empty, we return a product of 1.
        
        Args:
            seq_batch (SequenceBatch): of shape (batch_size, seq_length, X1, X2, ...)

        Returns:
            Tensor: of shape (batch_size, X1, X2, ...)
        """
        mask = seq_batch.mask
        values = seq_batch.values

        # We set all pad values = 1, so that taking the log will not produce -inf
        mask_bcast = expand_dims_for_broadcast(mask, values).expand(values.size())  # (batch_size, seq_length, X1, X2, ...)
        values = conditional(mask_bcast, values, 1 - mask_bcast)

        logged = SequenceBatch(torch.log(values), seq_batch.mask)  # (batch_size, seq_length, X1, X2, ...)

        log_sum = SequenceBatch.reduce_sum(logged)  # (batch_size, X1, X2, ...)
        prod = torch.exp(log_sum)
        return prod

    @classmethod
    def reduce_mean(cls, seq_batch, allow_empty=False):
        """Compute the mean of each sequence in a SequenceBatch.

        Args:
            seq_batch (SequenceBatch): a SequenceBatch with the following attributes:
                values (Tensor): a Tensor of shape (batch_size, seq_length, X1, X2, ...)
                mask (Tensor): if the mask values are arbitrary floats (rather than binary), the mean will be
                a weighted average.
            allow_empty (bool): allow computing the average of an empty sequence. In this case, we assume 0/0 == 0, rather
                than NaN. Default is False, causing an error to be thrown.

        Returns:
            Tensor: of shape (batch_size, X1, X2, ...)
        """
        values, mask = seq_batch.values, seq_batch.mask
        # compute weights for the average
        sums = torch.sum(mask, dim=1)  # (batch_size, 1)

        if allow_empty:
            sums[sums == 0.0] = 1.0  # Modify in-place: replace zeros with ones
        else:
            if (sums.data == 0).any():
                raise ValueError("Averaging zero elements.")

        weights = mask / sums.expand(*mask.size())
        return cls.weighted_sum(seq_batch, weights)

    @classmethod
    def _empty_seqs(cls, seq_batch):
        return (torch.sum(seq_batch.mask, 1).data == 0).any()

    @classmethod
    def reduce_max(cls, seq_batch):
        if cls._empty_seqs(seq_batch):
            raise ValueError("Taking max over zero elements.")
        values, mask = seq_batch.values, seq_batch.mask

        inf_mask = mask.clone()  # (batch_size, seq_length)
        inf_mask[mask == 0] = float('inf')
        inf_mask[mask == 1] = 0
        # masked elements will never win the max, because we subtract infinity from them

        inf_mask_bcast = expand_dims_for_broadcast(inf_mask, values).expand_as(values)  # (batch_size, seq_length, X1, X2, ...)

        max_values, _ = torch.max(values - inf_mask_bcast, 1)  # (batch_size, 1, X1, X2, ...)
        max_values = torch.squeeze(max_values, 1)  # (batch_size, X1, X2, ...)

        return max_values

    @classmethod
    def embed(cls, indices, embeds):
        """Embed a SequenceBatch of integers.
        
        Args:
            indices (SequenceBatch): of shape (batch_size, seq_length), with seq_batch.values of type LongTensor (ints)
            embeds (Variable): of shape (vocab_size, embed_dim)

        Returns:
            SequenceBatch: of shape (batch_size, seq_length, embed_dim)
        """
        values, mask = indices

        batch_size, seq_length = values.size()
        vocab_size, embed_dim = embeds.size()

        indices_flat = values.view(batch_size * seq_length)
        embedded_indices_flat = torch.index_select(embeds, 0, indices_flat)  # (batch_size * seq_length, embed_dim)
        embedded_indices = embedded_indices_flat.view(batch_size, seq_length, embed_dim)
        return SequenceBatch(embedded_indices, mask)


SequenceBatchElement = namedtuple('SequenceBatchElement', ['values', 'mask'])
