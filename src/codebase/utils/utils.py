import pdb
import random 
import numpy as np

from torch import _TensorBase, torch
from torch.autograd import Variable

from contextlib import contextmanager

def tile_state(h, batch_size):
    '''
    Tile a given hidden state_batch_size times.

    Ins:
        h (Variable): a single hidden state of shape (hidden_dim)
        batch_size (int)

    Outs:
        Variable of shape (batch_size, hidden_dim)
    '''
    return h.unsqueeze(0).repeat(*([batch_size] + [1] * len(h.size())))

def gated_update(h, h_new, update):
    """If update == 1.0, return h_new; if update == 0.0, return h.

    Applies this logic to each element in a batch.

    Args:
        h (Variable): of shape (batch_size, hidden_dim)
        h_new (Variable): of shape (batch_size, hidden_dim)
        update (Variable): of shape (batch_size, 1).

    Returns:
        Variable: of shape (batch_size, hidden_dim)

    """
    batch_size, hidden_dim = h.size()
    gate = update.expand(batch_size, hidden_dim)
    return conditional(gate, h_new, h)

def conditional(b, x, y):
    """Conditional operator for PyTorch.

    Args:
        b (FloatTensor): with values that are equal to 0 or 1
        x (FloatTensor): of same shape as b
        y (FloatTensor): of same shape as b

    Returns:
        z (FloatTensor): of same shape as b. z[i] = x[i] if b[i] == 1 else y[i]
    """
    return b * x + (1 - b) * y

def to_numpy(x):
    if isinstance(x, Variable):
        x = x.data  # unwrap Variable

    if isinstance(x, _TensorBase):
        x = x.cpu().numpy()
    return x

def GPUVariable(data):
    return try_gpu(Variable(data, requires_grad=False))

_GPUS_EXIST = True

def try_gpu(x):
    """Try to put a Variable/Tensor/Module on GPU."""
    global _GPUS_EXIST

    if _GPUS_EXIST:
        try:
            return x.cuda()
        except (AssertionError, RuntimeError):
            # actually, GPUs don't exist
            print('No GPUs detected. Sticking with CPUs.')
            _GPUS_EXIST = False
            return x
    else:
        return x

# necessary for Kelvin's vocabulary object...
class EqualityMixin(object):
    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

@contextmanager
def random_seed(seed=None):
    """Execute code inside this with-block using the specified seed.

    If no seed is specified, nothing happens.

    Does not affect the state of the random number generator outside this block.
    Not thread-safe.

    Args:
        seed (int): random seed
    """
    if seed is None:
        yield
    else:
        py_state = random.getstate()  # save state
        np_state = np.random.get_state()

        random.seed(seed)  # alter state
        np.random.seed(seed)
        yield

        random.setstate(py_state)  # restore state
        np.random.set_state(np_state)

def expand_dims_for_broadcast(low_tensor, high_tensor):
    """Expand the dimensions of a lower-rank tensor, so that its rank matches that of a higher-rank tensor.

    This makes it possible to perform broadcast operations between low_tensor and high_tensor.

    Args:
        low_tensor (Tensor): lower-rank Tensor with shape [s_0, ..., s_p]
        high_tensor (Tensor): higher-rank Tensor with shape [s_0, ..., s_p, ..., s_n]

    Note that the shape of low_tensor must be a prefix of the shape of high_tensor.

    Returns:
        Tensor: the lower-rank tensor, but with shape expanded to be [s_0, ..., s_p, 1, 1, ..., 1]
    """
    low_size, high_size = low_tensor.size(), high_tensor.size()
    low_rank, high_rank = len(low_size), len(high_size)

    # verify that low_tensor shape is prefix of high_tensor shape
    assert low_size == high_size[:low_rank]

    new_tensor = low_tensor
    for _ in range(high_rank - low_rank):
        new_tensor = torch.unsqueeze(new_tensor, len(new_tensor.size()))

    return new_tensor

def is_binary(t):
    """Check if values of t are binary.

    Args:
        t (Tensor|Variable)

    Returns:
        bool
    """
    if isinstance(t, Variable):
        t = t.data

    binary = (t == 0) | (t == 1)
    all_binary = torch.prod(binary)
    return all_binary == 1

class NamedTupleLike(object):
    __slots__ = []
