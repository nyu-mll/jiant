# Retokenization helpers.
# Use this to project span annotations from one tokenization to another.
#
# NOTE: please don't make this depend on any other jiant/ libraries; it would
# be nice to opensource this as a standalone utility.
#
# Current implementation is not fast; TODO to profile this and see why.

import functools
import re
from io import StringIO
from typing import Iterable, List, NewType, Sequence, Text, Tuple, Type, Union

import numpy as np
from nltk.tokenize.simple import SpaceTokenizer
from scipy import sparse

# Use https://pypi.org/project/python-Levenshtein/ for fast alignment.
# install with: pip install python-Levenshtein
from Levenshtein.StringMatcher import StringMatcher

from jiant.utils.tokenizers import get_tokenizer, Tokenizer
from jiant.utils.utils import unescape_moses


# Tokenizer instance for internal use.
_SIMPLE_TOKENIZER = SpaceTokenizer()
_SEP = " "  # should match separator used by _SIMPLE_TOKENIZER

# Type alias for internal matricies
Matrix = NewType("Matrix", Union[Type[sparse.csr_matrix], Type[np.ndarray]])

_DTYPE = np.int32


def _mat_from_blocks_dense(mb, n_chars_src, n_chars_tgt):
    M = np.zeros((n_chars_src, n_chars_tgt), dtype=_DTYPE)
    for i in range(len(mb)):
        b = mb[i]  # current block
        # Fill in-between this block and last block
        if i > 0:
            lb = mb[i - 1]  # last block
            s0 = lb[0] + lb[2]  # top
            e0 = b[0]  # bottom
            s1 = lb[1] + lb[2]  # left
            e1 = b[1]  # right
            M[s0:e0, s1:e1] = 1
        # Fill matching region on diagonal
        M[b[0] : b[0] + b[2], b[1] : b[1] + b[2]] = 2 * np.identity(b[2], dtype=_DTYPE)
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
            l0 = b[0] - s0  # num rows
            s1 = lb[1] + lb[2]  # left
            l1 = b[1] - s1  # num cols
            idxs = np.indices((l0, l1))
            ridxs.extend((s0 + idxs[0]).flatten())  # row indices
            cidxs.extend((s1 + idxs[1]).flatten())  # col indices
            data.extend(np.ones(l0 * l1, dtype=_DTYPE))

        # Fill matching region on diagonal
        ridxs.extend(range(b[0], b[0] + b[2]))
        cidxs.extend(range(b[1], b[1] + b[2]))
        data.extend(2 * np.ones(b[2], dtype=_DTYPE))
    M = sparse.csr_matrix((data, (ridxs, cidxs)), shape=(n_chars_src, n_chars_tgt))
    return M


def _mat_from_spans_dense(spans: Sequence[Tuple[int, int]], n_chars: int) -> Matrix:
    """Construct a token-to-char matrix from a list of char spans."""
    M = np.zeros((len(spans), n_chars), dtype=_DTYPE)
    for i, s in enumerate(spans):
        M[i, s[0] : s[1]] = 1
    return M


def _mat_from_spans_sparse(spans: Sequence[Tuple[int, int]], n_chars: int) -> Matrix:
    """Construct a token-to-char matrix from a list of char spans."""
    ridxs = []
    cidxs = []
    for i, s in enumerate(spans):
        ridxs.extend([i] * (s[1] - s[0]))  # repeat token index
        cidxs.extend(range(s[0], s[1]))  # char indices
        #  assert len(ridxs) == len(cidxs)
    data = np.ones(len(ridxs), dtype=_DTYPE)
    return sparse.csr_matrix((data, (ridxs, cidxs)), shape=(len(spans), n_chars))


def create_tokenization_alignment(
    tokens: Sequence[str], tokenizer_name: str
) -> Sequence[Tuple[str, str]]:
    """
    Builds alignment mapping between space tokenization and tokenization of 
    choice. 
    
    Example:
        Input: ['Larger', 'than', 'life.']
        Output: [('Larger', ['ĠL', 'arger']), ('than', ['Ġthan']), ('life.', ['Ġlife', '.'])]

    Parameters
    -----------------------
        tokens: list[(str)]. list of tokens, 
        tokenizer_name: str

    Returns
    -----------------------
        tokenization_mapping: list[(str, str)], list of tuples with (orig_token, tokenized_token).

    """
    tokenizer = get_tokenizer(tokenizer_name)
    tokenization_mapping = []
    for tok in tokens:
        aligned_tok = tokenizer.tokenize(tok)
        tokenization_mapping.append((tok, aligned_tok))
    return tokenization_mapping


def realign_spans(record, tokenizer_name):
    """
    Builds the indices alignment while also tokenizing the input
    piece by piece.
    Currently, SentencePiece (for XLNet), WPM (for BERT), BPE (for GPT/XLM),
    ByteBPE (for RoBERTa/GPT-2) and Moses (for Transformer-XL and default) tokenization are
    supported.

    Parameters
    -----------------------
        record: dict with the below fields
            text: str
            targets: list of dictionaries
                label: bool
                span1_index: int, start index of first span
                span1_text: str, text of first span
                span2_index: int, start index of second span
                span2_text: str, text of second span
        tokenizer_name: str

    Returns
    ------------------------
        record: dict with the below fields:
            text: str in tokenized form
            targets: dictionary with the below fields
                -label: bool
                -span_1: (int, int) of token indices
                -span1_text: str, the string
                -span2: (int, int) of token indices
                -span2_text: str, the string
    """

    # find span indices and text
    text = record["text"].split()
    span1 = record["target"]["span1_index"]
    span1_text = record["target"]["span1_text"]
    span2 = record["target"]["span2_index"]
    span2_text = record["target"]["span2_text"]

    # construct end spans given span text space-tokenized length
    span1 = [span1, span1 + len(span1_text.strip().split())]
    span2 = [span2, span2 + len(span2_text.strip().split())]
    indices = [span1, span2]

    sorted_indices = sorted(indices, key=lambda x: x[0])
    current_tokenization = []
    span_mapping = {}

    # align first span to tokenized text
    aligner_fn = get_aligner_fn(tokenizer_name)
    _, new_tokens = aligner_fn(" ".join(text[: sorted_indices[0][0]]))
    current_tokenization.extend(new_tokens)
    new_span1start = len(current_tokenization)
    _, span_tokens = aligner_fn(" ".join(text[sorted_indices[0][0] : sorted_indices[0][1]]))
    current_tokenization.extend(span_tokens)
    new_span1end = len(current_tokenization)
    span_mapping[sorted_indices[0][0]] = [new_span1start, new_span1end]

    # re-indexing second span
    _, new_tokens = aligner_fn(" ".join(text[sorted_indices[0][1] : sorted_indices[1][0]]))
    current_tokenization.extend(new_tokens)
    new_span2start = len(current_tokenization)
    _, span_tokens = aligner_fn(" ".join(text[sorted_indices[1][0] : sorted_indices[1][1]]))
    current_tokenization.extend(span_tokens)
    new_span2end = len(current_tokenization)
    span_mapping[sorted_indices[1][0]] = [new_span2start, new_span2end]

    # save back into record
    _, all_text = aligner_fn(" ".join(text))
    record["target"]["span1"] = span_mapping[record["target"]["span1_index"]]
    record["target"]["span2"] = span_mapping[record["target"]["span2_index"]]
    record["text"] = " ".join(all_text)
    return record


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

    def _mat_from_blocks(
        self, mb: Sequence[Tuple[int, int, int]], n_chars_src: int, n_chars_tgt: int
    ) -> Matrix:
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

    def __init__(self, source: Union[Iterable[str], str], target: Union[Iterable[str], str]):
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
        output.write("{:s}({:d}, {:d}):\n".format(self.__class__.__name__, *self.T.shape))
        for i in range(self.T.shape[0]):
            targs = sorted(list(self.project_tokens(i)))
            output.write("  {:d} -> {:s}".format(i, str(targs)))
            if src_tokens is not None and tgt_tokens is not None:
                tgt_list = [tgt_tokens[j] for j in targs]
                output.write("\t'{:s}' -> {:s}".format(src_tokens[i], str(tgt_list)))
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


##
# Aligner functions. These take a raw string and return a tuple
# of a TokenAligner instance and a list of tokens.
##


def space_tokenize_with_eow(sentence):
    """Add </w> markers to ensure word-boundary alignment."""
    return [t + "</w>" for t in sentence.split()]


def process_wordpiece_for_alignment(t):
    """Add <w> markers to ensure word-boundary alignment."""
    if t.startswith("##"):
        return re.sub(r"^##", "", t)
    else:
        return "<w>" + t


def process_sentencepiece_for_alignment(t):
    """Add <w> markers to ensure word-boundary alignment."""
    if t.startswith("▁"):
        return "<w>" + re.sub(r"^▁", "", t)
    else:
        return t


def process_bytebpe_for_alignment(t):
    """Add <w> markers to ensure word-boundary alignment."""
    if t.startswith("▁"):
        return "<w>" + re.sub(r"^Ġ", "", t)
    else:
        return t


def space_tokenize_with_bow(sentence):
    """Add <w> markers to ensure word-boundary alignment."""
    return ["<w>" + t for t in sentence.split()]


def align_moses(text: Text) -> Tuple[TokenAligner, List[Text]]:
    """Aligner fn for Moses tokenizer, used in Transformer-XL
    """
    MosesTokenizer = get_tokenizer("MosesTokenizer")
    moses_tokens = MosesTokenizer.tokenize(text)
    cleaned_moses_tokens = unescape_moses(moses_tokens)
    ta = TokenAligner(text, cleaned_moses_tokens)
    return ta, moses_tokens


def align_wpm(
    text: Text, wpm_tokenizer: Tokenizer, do_lower_case: bool
) -> Tuple[TokenAligner, List[Text]]:
    """Alignment fn for WPM tokenizer, used in BERT
    """
    # If using lowercase, do this for the source tokens for better matching.
    bow_tokens = space_tokenize_with_bow(text.lower() if do_lower_case else text)
    wpm_tokens = wpm_tokenizer.tokenize(text)

    # Align using <w> markers for stability w.r.t. word boundaries.
    modified_wpm_tokens = list(map(process_wordpiece_for_alignment, wpm_tokens))
    ta = TokenAligner(bow_tokens, modified_wpm_tokens)
    return ta, wpm_tokens


def align_sentencepiece(
    text: Text, sentencepiece_tokenizer: Tokenizer
) -> Tuple[TokenAligner, List[Text]]:
    """Alignment fn for SentencePiece Tokenizer, used in XLNET
    """
    bow_tokens = space_tokenize_with_bow(text)
    sentencepiece_tokens = sentencepiece_tokenizer.tokenize(text)

    modified_sentencepiece_tokens = list(
        map(process_sentencepiece_for_alignment, sentencepiece_tokens)
    )
    ta = TokenAligner(bow_tokens, modified_sentencepiece_tokens)
    return ta, sentencepiece_tokens


def align_bpe(text: Text, bpe_tokenizer: Tokenizer) -> Tuple[TokenAligner, List[Text]]:
    """Alignment fn for BPE tokenizer, used in GPT and XLM
    """
    eow_tokens = space_tokenize_with_eow(text.lower())
    bpe_tokens = bpe_tokenizer.tokenize(text)
    ta = TokenAligner(eow_tokens, bpe_tokens)
    return ta, bpe_tokens


def align_bytebpe(text: Text, bytebpe_tokenizer: Tokenizer) -> Tuple[TokenAligner, List[Text]]:
    """Alignment fn for Byte-level BPE tokenizer, used in GPT-2 and RoBERTa
    """
    bow_tokens = space_tokenize_with_bow(text)
    bytebpe_tokens = bytebpe_tokenizer.tokenize(text)

    modified_bytebpe_tokens = list(map(process_bytebpe_for_alignment, bytebpe_tokens))
    ta = TokenAligner(bow_tokens, modified_bytebpe_tokens)
    return ta, bytebpe_tokens


def get_aligner_fn(tokenizer_name: Text):
    """Given the tokenzier_name, return the corresponding alignment function.
    An alignment function modified the tokenized input to make it close to source token,
    and choose a space tokenizer with its word-boundary at the same side as tokenizer_name,
    hence the source (from space tokenizer) and target (from tokenizer_name) is sufficiently close.
    Use TokenAligner to project token index from source to target.
    """
    if tokenizer_name == "MosesTokenizer" or tokenizer_name.startswith("transfo-xl-"):
        return align_moses
    elif tokenizer_name.startswith("bert-"):
        do_lower_case = tokenizer_name.endswith("uncased")
        wpm_tokenizer = get_tokenizer(tokenizer_name)
        return functools.partial(
            align_wpm, wpm_tokenizer=wpm_tokenizer, do_lower_case=do_lower_case
        )
    elif tokenizer_name.startswith("openai-gpt") or tokenizer_name.startswith("xlm-mlm-en-"):
        bpe_tokenizer = get_tokenizer(tokenizer_name)
        return functools.partial(align_bpe, bpe_tokenizer=bpe_tokenizer)
    elif tokenizer_name.startswith("xlnet-"):
        sentencepiece_tokenizer = get_tokenizer(tokenizer_name)
        return functools.partial(
            align_sentencepiece, sentencepiece_tokenizer=sentencepiece_tokenizer
        )
    elif tokenizer_name.startswith("roberta-") or tokenizer_name.startswith("gpt2"):
        bytebpe_tokenizer = get_tokenizer(tokenizer_name)
        return functools.partial(align_bytebpe, bytebpe_tokenizer=bytebpe_tokenizer)
    else:
        raise ValueError(f"Unsupported tokenizer '{tokenizer_name}'")
