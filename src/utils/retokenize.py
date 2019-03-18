# Retokenization helpers.
# Use this to project span annotations from one tokenization to another.
#
# NOTE: please don't make this depend on any other jiant/ libraries; it would
# be nice to opensource this as a standalone utility.
#
# Current implementation is not fast; TODO to profile this and see why.

from typing import Sequence, Iterable, Tuple, \
    Union, Type, NewType

from io import StringIO

import numpy as np
from scipy import sparse
from ..utils import utils
import re
from nltk.tokenize.simple import SpaceTokenizer

# Use https://pypi.org/project/python-Levenshtein/ for fast alignment.
# install with: pip install python-Levenshtein
from Levenshtein.StringMatcher import StringMatcher
from typing import Tuple, List, Text
# Tokenizer instance for internal use.
_SIMPLE_TOKENIZER = SpaceTokenizer()
_SEP = " "  # should match separator used by _SIMPLE_TOKENIZER

def align_tags(orig_text, current_tags, introduce_tag):
    """
    Align tag labels from source tokenization to target tokenization.
    Parameters
    -------------
    text: List[str] the text tokenized in target tokenization
    orig_text: the original text input pre-tokenization
    alignment_tokenizer_func: Func that returns TokenAligner between two tokenizations
    current_tags: List of tags in source tokenization
    Returns
    -------------
    List of tags in target tokenization
    """
    import pdb; pdb.set_trace()
    # remove apos&  introduced by Moses Tokenization,apos is 42 in vocab.
    text = text[1:-1]
    ta, _ = align_bert(orig_text, model_name="bert-base-uncased")
    current_indices = [(i, i+1) for i in range(len(current_tags))]
    new_tag_indices = []
    import pdb; pdb.set_trace()
    for tag_index in current_indices:
        new_tag_indices.append(ta.project_span(tag_index[0], tag_index[1]))
    aligned_tags = []
    for i in range(len(new_tag_indices)):
        new_start = new_tag_indices[i][0]
        new_end = new_tag_indices[i][1]
        aligned_tags.append(current_tags[i])
        if (new_end - new_start) > 1:
            for i in new_end - new_start - 1:
                aligned_tags.append(introduce_tag)
    assert len(text) == len(aligned_tags)
    res_aligned_tags = [str(int(tag) + utils.EOS_INDEX + 1) for tag in aligned_tags]
    res_aligned_tags = [str(utils.SOS_INDEX)] + res_aligned_tags + [str(utils.EOS_INDEX)]
    return res_aligned_tags


def adjust_targs_moses_to_BPE(targs, bpe_input, orig_input, orig_tokenizer_func):
    """
    Adjusts target tokens so that it matches with the newly BPE-retokenized
    input
    This function is called only for OpenAI -> TaggingTasks
    BPE tokenization will add more tokens to the sentence than the original
    tokens (thus len(bpe_input) >= len(orig_input)).
    However, MosesTokenizer can add ;apos to the text, thus we first must remove
    the tags correlating to &apos; as well as the &apos; in the Moses-tokenized
    text.
    We handle aligning by assuming that the target token of the
    new added bpe token is the same as its left neighbor for simplicity.
    Parameters
    ----------
    targs :  A list of string, required.
        target tokens
    bpe_input : list of string, required
        This is the output of the BPE tokenizer
    orig_input: string, required
        This is the original input from the dataset
    orig_tokenizer_func: func: str -> list of strings
        This is a function that tokenizes it the way that the dataset is
        tokenized. For example, CCGBank was created by tokenizing the input
        using MosesTokenizer
    Returns
    -------
    A list of strings of the newly adjusted target tokens
    """
    orig_input = orig_tokenizer_func(orig_input)
    indices = [i[0] for i in enumerate(orig_input) if i[1] == '&apos;']
    for index in sorted(indices, reverse=True):
        del targs[index]
    orig_input = [tok for tok in orig_input if tok != '&apos;']
    orig_toks = [s.lower() for s in orig_input]

    bpe_input = bpe_input[1:-1]
    new_toks = [s.replace("</w>", "") for s in bpe_input]

    c_index = len(targs) - 1
    curr_targ = targs[c_index]
    next_expected_tok = orig_toks[c_index-1]
    next_expected_tok = "".join(re.findall("[a-zA-Z]+", next_expected_tok))
    new_targs = [curr_targ]
    current_tok_up_to_now = ""
    # move from last to front
    for i in range(len(new_toks) - 2, -1, -1):
        # if the levenshtein distance is similar enough, then create it 
        current_tok_up_to_now = "%s%s" % (new_toks[i], current_tok_up_to_now)
        current_tok_up_to_now = "".join(re.findall("[a-zA-Z]+", current_tok_up_to_now))
        if current_tok_up_to_now != next_expected_tok:
            new_targs.append(curr_targ)
        else:
            c_index -= 1
            current_tok_up_to_now = ""
            curr_targ = targs[c_index]
            new_targs.append(curr_targ)
            next_expected_tok = orig_toks[c_index-1]
            next_expected_tok = "".join(re.findall("[a-zA-Z]+", next_expected_tok))
    # add EOS/SOS to the target tokens.
    targ_final = [str(int(t)+utils.EOS_INDEX+1) for t in new_targs]
    targ_final = [str(utils.EOS_INDEX)] + targ_final + [str(utils.SOS_INDEX)]
    targ_final.reverse()
    return targ_final

# Type alias for internal matricies
Matrix = NewType("Matrix", Union[Type[sparse.csr_matrix],
                                 Type[np.ndarray]])

_DTYPE = np.int32

def _mat_from_blocks_dense(mb, n_chars_src, n_chars_tgt):
    M = np.zeros((n_chars_src, n_chars_tgt), dtype=_DTYPE)
    for i in range(len(mb)):
        b = mb[i]  # current block
        # Fill in-between this block and last block
        if i > 0:
            lb = mb[i - 1]  # last block
            s0 = lb[0] + lb[2]  # top
            e0 = b[0]         # bottom
            s1 = lb[1] + lb[2]  # left
            e1 = b[1]         # right
            M[s0:e0, s1:e1] = 1
        # Fill matching region on diagonal
        M[b[0]:b[0] + b[2], b[1]:b[1] + b[2]] = 2 * np.identity(b[2], dtype=_DTYPE)
    return M


def _mat_from_blocks_sparse(mb, n_chars_src, n_chars_tgt):
    ridxs = []
    cidxs = []
    data = []
    for i, b in enumerate(mb):
        # Fill in-between this block and last block
        if i > 0:
            lb = mb[i - 1]  # last block
            s0 = lb[0] + lb[2]  # top
            l0 = b[0] - s0    # num rows
            s1 = lb[1] + lb[2]  # left
            l1 = b[1] - s1    # num cols
            idxs = np.indices((l0, l1))
            ridxs.extend((s0 + idxs[0]).flatten())  # row indices
            cidxs.extend((s1 + idxs[1]).flatten())  # col indices
            data.extend(np.ones(l0 * l1, dtype=_DTYPE))

        # Fill matching region on diagonal
        ridxs.extend(range(b[0], b[0] + b[2]))
        cidxs.extend(range(b[1], b[1] + b[2]))
        data.extend(2 * np.ones(b[2], dtype=_DTYPE))
    M = sparse.csr_matrix((data, (ridxs, cidxs)),
                          shape=(n_chars_src, n_chars_tgt))
    return M


def _mat_from_spans_dense(spans: Sequence[Tuple[int, int]],
                          n_chars: int) -> Matrix:
    """Construct a token-to-char matrix from a list of char spans."""
    M = np.zeros((len(spans), n_chars), dtype=_DTYPE)
    for i, s in enumerate(spans):
        M[i, s[0]:s[1]] = 1
    return M


def _mat_from_spans_sparse(spans: Sequence[Tuple[int, int]],
                           n_chars: int) -> Matrix:
    """Construct a token-to-char matrix from a list of char spans."""
    ridxs = []
    cidxs = []
    for i, s in enumerate(spans):
        ridxs.extend([i] * (s[1] - s[0]))  # repeat token index
        cidxs.extend(range(s[0], s[1]))    # char indices
        #  assert len(ridxs) == len(cidxs)
    data = np.ones(len(ridxs), dtype=_DTYPE)
    return sparse.csr_matrix((data, (ridxs, cidxs)),
                             shape=(len(spans), n_chars))


class TokenAligner(object):
    """Align two similiar tokenizations.

    Usage:
        ta = TokenAligner(source_tokens, target_tokens)
        target_span = ta.project_span(*source_span)

    Uses Levenshtein distance to align two similar tokenizations, and provide
    facilities to project indices and spans from the source tokenization to the
    target.

    Let source contain m tokens and M chars, and target contain n tokens and N
    chars. The token alignment is treated as a (sparse) m x n adjacency matrix
    T representing the bipartite graph between the source and target tokens.

    This is constructed by performing a character-level alignment using
    Levenshtein distance to obtain a (M x N) character adjacency matrix C. We
    then construct token-to-character matricies U (m x M) and V (n x N) and
    construct T as:
        T = (U C V')
    where V' denotes the transpose.

    Spans of non-aligned bytes are assumed to contain a many-to-many alignment
    of all chars in that range. This can lead to unwanted alignments if, for
    example, two consecutive tokens are mapped to escape sequences:
        source: ["'s", "["]
        target: ["&apos;", "s", "&#91;"]
    In the above case, "'s'" may be spuriously aligned to "&apos;" while "["
    has no character match with "s" or "&#91;", and so is aligned to both. To
    make a correct alignment more likely, ensure that the characters in target
    correspond as closely as possible to those in source. For example, the
    following will align correctly:
        source: ["'s", "["]
        target: ["'", "s", "["]
    """

    def token_to_char(self, text: str) -> Matrix:
        spans = _SIMPLE_TOKENIZER.span_tokenize(text)
        # TODO(iftenney): compare performance of these implementations.
        return _mat_from_spans_sparse(tuple(spans), len(text))
        #  return _mat_from_spans_dense(tuple(spans), len(text))

    def _mat_from_blocks(self, mb: Sequence[Tuple[int, int, int]],
                         n_chars_src: int, n_chars_tgt: int) -> Matrix:
        """Construct a char-to-char matrix from a list of matching blocks.

        mb is a sequence of (s1, s2, n_char) tuples, where s1 and s2 are the
        start indices in the first and second sequence, and n_char is the
        length of the block.
        """
        # TODO(iftenney): compare performance of these implementations.
        #  return _mat_from_blocks_sparse(mb, n_chars_src, n_chars_tgt)
        return _mat_from_blocks_dense(mb, n_chars_src, n_chars_tgt)

    def char_to_char(self, source: str, target: str) -> Matrix:
        # Run Levenshtein at character level.
        sm = StringMatcher(seq1=source, seq2=target)
        mb = sm.get_matching_blocks()
        return self._mat_from_blocks(mb, len(source), len(target))

    def __init__(self,
                 source: Union[Iterable[str], str],
                 target: Union[Iterable[str], str]):
        # Coerce source and target to space-delimited string.
        if not isinstance(source, str):
            source = _SEP.join(source)
        if not isinstance(target, str):
            target = _SEP.join(target)

        self.U = self.token_to_char(source)  # (M x m)
        self.V = self.token_to_char(target)  # (N x n)
        # Character transfer matrix (M x N)
        self.C = self.char_to_char(source, target)
        # Token transfer matrix (m x n)
        self.T = self.U * self.C * self.V.T
        #  self.T = self.U.dot(self.C).dot(self.V.T)

    def __str__(self):
        return self.pprint()

    def pprint(self, src_tokens=None, tgt_tokens=None) -> str:
        """Render as alignment table: src -> [tgts]"""
        output = StringIO()
        output.write("{:s}({:d}, {:d}):\n".format(self.__class__.__name__,
                                                  *self.T.shape))
        for i in range(self.T.shape[0]):
            targs = sorted(list(self.project_tokens(i)))
            output.write("  {:d} -> {:s}".format(i, str(targs)))
            if src_tokens is not None and tgt_tokens is not None:
                tgt_list = [tgt_tokens[j] for j in targs]
                output.write("\t'{:s}' -> {:s}".format(src_tokens[i],
                                                       str(tgt_list)))
            output.write("\n")
        return output.getvalue()

    def project_tokens(self, idxs: Union[int, Sequence[int]]) -> Sequence[int]:
        """Project source token indices to target token indices."""
        if isinstance(idxs, int):
            idxs = [idxs]
        return self.T[idxs].nonzero()[1]  # column indices

    def project_span(self, start, end) -> Tuple[int, int]:
        """Project a span from source to target.

        Span end is taken to be exclusive, so this actually projects end - 1
        and maps back to an exclusive target span.
        """
        tgt_idxs = self.project_tokens([start, end - 1])
        return min(tgt_idxs), max(tgt_idxs) + 1
