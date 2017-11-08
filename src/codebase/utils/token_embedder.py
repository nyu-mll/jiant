import pdb
import codecs
from abc import ABCMeta, abstractmethod
from collections import Mapping

import torch
import numpy as np
from codebase.utils.utils import GPUVariable
from codebase.utils.utils import EqualityMixin, random_seed
from codebase.utils.seq_batch import SequenceBatch
from torch.nn import Embedding, Module


class TokenEmbedder(Module):
    """
    Attributes:
        vocab (WordVocab)
        embeds (Variable): of shape (vocab_size, embed_dim)
        embed_dim (int)
    """

    def __init__(self, embeddings, vocab, trainable=True):
        """Create TokenEmbedder.

        Args:
            word_embeddings (WordEmbeddings)
            trainable (bool): if False, the embedding array will not see
                gradient steps
        """
        super(TokenEmbedder, self).__init__()

        arr = embeddings # np.ndarray
        vocab_size, embed_dim = arr.shape

        assert vocab_size == len(vocab)
        self.vocab = vocab
        self.embed_dim = embed_dim

        # create Embedding Module
        vocab_size, embed_dim = arr.shape
        self._embedding = TrainFlagEmbedding(
            vocab_size, embed_dim, arr, trainable=trainable)

    @property
    def embeds(self):
        """Return Variable of shape (vocab_size, embed_dim)."""
        return self._embedding.weight

    def embed_indices(self, indices):
        """Embed array of indices.

        Args:
            indices (Variable[LongTensor]): of shape (X1, X2) or (X1)

        Returns:
            embeds (Variable[FloatTensor]): of shape (X1, X2, embed_dim) or (X1, embed_dim)
        """
        return self._embedding(indices)

    def embed_seq_batch(self, seq_batch):
        """Embed elements of a SequenceBatch.

        Args:
            seq_batch (SequenceBatch)

        Returns:
            SequenceBatch
        """
        return SequenceBatch(self._embedding(seq_batch.values), seq_batch.mask)

    def embed_tokens(self, tokens):
        """Embed list of tokens.

        Args:
            tokens (list[unicode])

        Returns:
            embeds (Variable[FloatTensor]): of shape (len(tokens), embed_dim)
        """
        vocab = self.vocab
        indices = GPUVariable(torch.LongTensor([vocab.word2index(t) for t in tokens]))
        return self._embedding(indices)


class TrainFlagEmbedding(Module):
    """Small wrapper around PyTorch Embedding object. Exports a trainable
    flag, which allows you to fix the weights matrix. Obeys same interface as
    PyTorch Embedding object, except for extra constructor arguments.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 initial_embeddings, **kwargs):
        """Constructs TrainFlagEmbedding with embeddings initialized with
        initial_embeddings.

        Args:
            num_embeddings (int)
            embedding_dim (int)
            initial_embeddings (np.array): (num_embeddings, embedding_dim)
            trainable (bool): if False, weights matrix will not change.
                (default True)
            kwargs: all other supported keywords in torch.nn.Embeddings.
        """
        super(TrainFlagEmbedding, self).__init__()
        trainable = kwargs.pop("trainable", True)
        self._trainable = trainable
        if trainable:
            embedding = Embedding(
                num_embeddings, embedding_dim, **kwargs)
            embedding.weight.data.set_(
                torch.from_numpy(initial_embeddings).float())
            self._embedding = embedding
            self._weight = embedding.weight
        else:
            self._weight = GPUVariable(
                torch.from_numpy(initial_embeddings))

    @property
    def weight(self):
        return self._weight

    def forward(self, index):
        """Looks up a batch of indices.

        Args:
            index (Variable[LongTensor]): (batch, indices per batch)

        Returns:
            Tensor: (batch, indices per batch, embedding_dim)
        """
        if self._trainable:
            return self._embedding(index)
        else:
            batch, num_indices = index.size()
            flattened_index = index.view(batch * num_indices)
            embeddings = torch.index_select(
                self._weight, 0, flattened_index)
            return embeddings.view(batch, num_indices, -1)

class Vocab(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def word2index(self, w):
        pass

    @abstractmethod
    def index2word(self, i):
        pass


class SimpleVocab(Vocab, EqualityMixin):
    """A simple vocabulary object."""

    def __init__(self, tokens):
        """Create a vocab.

        Args:
            tokens (list[unicode]): a unique list of unicode tokens

        If t = tokens[i], this vocab will map token t to the integer i.
        """
        if not isinstance(tokens, list):
            raise ValueError('tokens must be a list')

        # build mapping
        word2index = {}
        for i, tok in enumerate(tokens):
            word2index[tok] = i

        if len(tokens) != len(word2index):
            raise ValueError('tokens must be unique')

        self._index2word = list(tokens)  # make a copy
        self._word2index = word2index

    @property
    def tokens(self):
        """Return the full list of tokens sorted by their index."""
        return self._index2word

    def __iter__(self):
        """Iterate through the full list of tokens."""
        return iter(self._index2word)

    def __len__(self):
        """Total number of tokens indexed."""
        return len(self._index2word)

    def __contains__(self, w):
        """Check if a token has been indexed by this vocab."""
        return w in self._word2index

    def word2index(self, w):
        return self._word2index[w]

    def index2word(self, i):
        return self._index2word[i]

    def words2indices(self, words):
        return map(self.word2index, words)

    def indices2words(self, indices):
        return [self.index2word(i) for i in indices]

    def save(self, path):
        """Save SimpleVocab to file path.

        Args:
            path (str)
        """
        with open(path, 'w') as f:
            for word in self._index2word:
                f.write(word)
                f.write('\n')

    @classmethod
    def load(cls, path):
        """Load SimpleVocab from file path.

        Args:
            path (str)

        Returns:
            SimpleVocab
        """
        strip_newline = lambda s: s[:-1]
        with open(path, 'r') as f:
            tokens = [strip_newline(line) for line in f]
        return cls(tokens)


class WordVocab(SimpleVocab):
    """WordVocab.

    IMPORTANT NOTE: WordVocab is blind to casing! All words are converted to lower-case.

    A WordVocab is required to have the following special tokens: UNK, START, STOP.
    """
    UNK = u'<unk>'
    START = u'<start>'
    STOP = u'<stop>'
    SPECIAL_TOKENS = (UNK, START, STOP)

    def __init__(self, tokens):
        super(WordVocab, self).__init__([t.lower() for t in tokens])

        # make sure all special tokens present
        for special in self.SPECIAL_TOKENS:
            if special not in self:
                raise ValueError('All special tokens must be present in tokens. Missing {}'.format(special))

    def word2index(self, w):
        """Map a word to an integer.

        Automatically lower-cases the word before mapping it.

        If the word is not known to the vocab, return the index for UNK.
        """
        sup = super(WordVocab, self)
        try:
            return sup.word2index(w.lower())
        except KeyError:
            return sup.word2index(self.UNK)


class SimpleEmbeddings(Mapping):
    def __init__(self, array, vocab):
        """Create embeddings object.

        Args:
            array (np.array): has shape (vocab_size, embed_dim)
            vocab (SimpleVocab): a Vocab object
        """
        assert len(array.shape) == 2
        assert array.shape[0] == len(vocab)  # entries line up

        self.array = array
        self.vocab = vocab

    def __contains__(self, w):
        return w in self.vocab

    def __getitem__(self, w):
        idx = self.vocab.word2index(w)
        return np.copy(self.array[idx])

    def __iter__(self):
        return iter(self.vocab)

    def __len__(self):
        return len(self.vocab)

    @property
    def embed_dim(self):
        return self.array.shape[1]

    @classmethod
    def from_file(cls, file_path, embed_dim, vocab_size=None):
        """Load word embeddings.

        Args:
            file_path (str)
            embed_dim (int): expected embed_dim
            vocab_size (int): max # of words in the vocab. If not specified, uses all available vectors in file.
        """
        if vocab_size is None:
            vocab_size = num_lines(file_path)

        words = []
        embeds = []
        with codecs.open(file_path, 'r', encoding='utf-8') as f:
            #lines = verboserate(f, desc='Loading embeddings from {}'.format(file_path), total=vocab_size)
            #for i, line in enumerate(lines):
            for i, line in enumerate(f):
                if i == vocab_size: break
                tokens = line.split()
                word, embed = tokens[0], np.array([float(tok) for tok in tokens[1:]], dtype=np.float32)
                if len(embed) != embed_dim:
                    raise ValueError('expected {} dims, got {} dims'.format(embed_dim, len(embed)))
                words.append(word)
                embeds.append(embed)

        vocab = SimpleVocab(words)
        embed_matrix = np.stack(embeds)
        embed_matrix = embed_matrix.astype(np.float32)
        assert embed_matrix.shape == (vocab_size, embed_dim)
        return cls(embed_matrix, vocab)

    def to_file(self, file_path):
        array = self.array
        with codecs.open(file_path, 'w', encoding='utf-8') as f:
            for i, word in enumerate(self.vocab):
                vec_str = u' '.join(str(x) for x in array[i])
                f.write(u'{} {}'.format(word, vec_str))
                f.write('\n')

    def with_special_tokens(self, random_seed=0):
        """Return a new SimpleEmbeddings object with special tokens inserted at the front of the vocab.
        
        In the new vocab, special tokens will occupy indices 0, 1, ..., len(special_tokens) - 1.
        The special tokens will have randomly generated embeddings.

        Args:
            random_seed (int)
        
        Returns:
            SimpleEmbeddings
        """
        special_tokens = list(WordVocab.SPECIAL_TOKENS)
        _, embed_dim = self.array.shape
        special_tokens_array_shape = (len(special_tokens), embed_dim)
        special_tokens_array = emulate_distribution(special_tokens_array_shape, self.array, seed=random_seed)
        special_tokens_array = special_tokens_array.astype(np.float32)

        new_array = np.concatenate((special_tokens_array, self.array), axis=0)
        new_vocab = WordVocab(special_tokens + self.vocab.tokens)

        return SimpleEmbeddings(new_array, new_vocab)


def emulate_distribution(shape, target_samples, seed=None):
    m = np.mean(target_samples)
    s = np.std(target_samples)

    with random_seed(seed):
        samples = np.random.normal(m, s, size=shape)

    return samples
