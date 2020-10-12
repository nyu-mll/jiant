"""Retokenization helpers

This module provides helpers for projecting span annotations from one tokenization to another.

Notes:
    * Code is ported from https://github.com/nyu-mll/jiant/blob/master/jiant/utils/retokenize.py
    * Please keep this code as a standalone utility; don't make this module depend on jiant modules.

"""
from typing import Iterable, Sequence, Tuple, Union

from Levenshtein.StringMatcher import StringMatcher
from nltk.tokenize.util import string_span_tokenize
import numpy as np


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


def _mat_from_spans_dense(spans: Sequence[Tuple[int, int]], n_chars: int) -> np.ndarray:
    """Construct a token-to-char matrix from a list of char spans."""
    M = np.zeros((len(spans), n_chars), dtype=_DTYPE)
    for i, s in enumerate(spans):
        M[i, s[0] : s[1]] = 1
    return M


def token_to_char(text: str, sep=" ") -> np.ndarray:
    """Takes a string, space tokenizes the string, and returns a mapping from tokens to chars.

    Examples:
        >>> token_to_char("testing 1, 2, 3")
        # produces a (m) token by (M) char matrix:

                   t e s t i n g   1 ,   2 ,   3
         testing [[1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
              1,  [0 0 0 0 0 0 0 0 1 1 0 0 0 0 0]
              2,  [0 0 0 0 0 0 0 0 0 0 0 1 1 0 0]
              3   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]]

    Args:
        text (str): string to tokenize and build the token to char mapping.

    Returns:
        np.ndarray mapping from (m) tokens to (M) chars.

    """
    spans = string_span_tokenize(text, sep=sep)
    return _mat_from_spans_dense(tuple(spans), len(text))


def _mat_from_blocks(
    mb: Sequence[Tuple[int, int, int]], n_chars_src: int, n_chars_tgt: int
) -> np.ndarray:
    """Construct a char-to-char matrix from a list of matching blocks.

    mb is a sequence of (s1, s2, n_char) tuples, where s1 and s2 are the start indices in the
    first and second sequence, and n_char is the length of the block.

    Args:
        mb: list of triples of non-overlapping matching subsequences in source str and target.
        n_chars_src (int): number of chars in the source string.
        n_chars_tgt (int): number of chars in the target string.

    Returns:
        np.ndarray adjacency matrix mapping chars in the source str to chars in the target str.

    """
    return _mat_from_blocks_dense(mb, n_chars_src, n_chars_tgt)


def char_to_char(source: str, target: str) -> np.ndarray:
    """Find the character adjacency matrix mapping source string chars to target string chars.

    Uses StringMatcher from Levenshtein package to find non-overlapping matching subsequences in
    input strings. Uses the result to create a character adjacency matrix from source to target.
    (https://docs.python.org/2/library/difflib.html#difflib.SequenceMatcher.get_matching_blocks)

    Args:
        source (str): string of source chars.
        target (str): string of target chars.

    Returns:
        np.ndarray adjacency matrix mapping chars in the source str to chars in the target str.

    """
    sm = StringMatcher(seq1=source, seq2=target)
    mb = sm.get_matching_blocks()
    return _mat_from_blocks(mb, len(source), len(target))


class TokenAligner(object):
    """Align two similiar tokenizations.

    Args:
        source (Union[Iterable[str], str]): Source text tokens or string.
        target (Union[Iterable[str], str]): Target text tokens or string.

    Usage:
        ta = TokenAligner(source_tokens, target_tokens)
        target_span = ta.project_span(*source_span)

    Uses Levenshtein distance to align two similar tokenizations, and provide facilities to project
    indices and spans from the source tokenization to the target.

    Let source contain m tokens and M chars, and target contain n tokens and N chars. The token
    alignment is treated as a (sparse) m x n adjacency matrix T representing the bipartite graph
    between the source and target tokens.

    This is constructed by performing a character-level alignment using Levenshtein distance to
    obtain a (M x N) character adjacency matrix C. We then construct token-to-character matricies
    U (m x M) and V (n x N) and construct T as:
        T = (U C V')
    where V' denotes the transpose.

    Spans of non-aligned bytes are assumed to contain a many-to-many alignment of all chars in that
    range. This can lead to unwanted alignments if, for example, two consecutive tokens are mapped
    to escape sequences:
        source: ["'s", "["]
        target: ["&apos;", "s", "&#91;"]
    In the above case, "'s'" may be spuriously aligned to "&apos;" while "[" has no character match
    with "s" or "&#91;", and so is aligned to both. To make a correct alignment more likely, ensure
    that the characters in target correspond as closely as possible to those in source. For example,
    the following will align correctly:
        source: ["'s", "["]
        target: ["'", "s", "["]

    """

    def __init__(self, source: Union[Iterable[str], str], target: Union[Iterable[str], str]):
        # Coerce source and target to space-delimited string.
        if not isinstance(source, str):
            source = " ".join(source)
        if not isinstance(target, str):
            target = " ".join(target)
        self.U = token_to_char(source)  # (m X M) source token idx to source char idx
        self.V = token_to_char(target)  # (n x N) target token idx to target char idx
        self.C = char_to_char(source, target)  # (M x N) source char idx to target char idx
        # Token transfer matrix from (m) tokens in source to (n) tokens in the target. Mat value at
        # index i, j measures the character overlap btwn the ith source token and jth target token.
        self.source_token_idx_to_target_token_idx = self.U.dot(self.C).dot(self.V.T)
        self.source_token_idx_to_target_char_idx = self.U.dot(self.C)
        self.source_char_idx_to_target_token_idx = self.C.dot(self.V.T)

    def project_token_idxs(self, idxs: Union[int, Sequence[int]]) -> Sequence[int]:
        """Project source token index(s) to target token indices.

        Takes a list of token indices in the source token sequence, and returns the corresponding
        tokens in the target sequence.

        Args:
            idxs (Union[int, Sequence[int]]): source token index(s) to get related target indices.

        Examples:
            >>> source_tokens = ['abc', 'def', 'ghi', 'jkl']
            >>> target_tokens = ['abc', 'd', 'ef', 'ghi', 'jkl']
            >>> ta = TokenAligner(source_tokens, target_tokens)
            >>> print(ta.project_token_idxs([1, 2]))
            [1 2 3]

        Returns:
            List[int] representing the target indices associated with the provided source indices.

        """
        if isinstance(idxs, int):
            idxs = [idxs]
        return self.source_token_idx_to_target_token_idx[idxs].nonzero()[1]  # column indices

    @staticmethod
    def _project_span(mat, start, end, inclusive):
        if inclusive:
            end = end + 1
        target_matches = mat[start:end].nonzero()[1].tolist()
        if len(target_matches) == 0:
            raise ValueError(f"Project {(start, end)} into empty span in target sequence")
        output_start, output_end = min(target_matches), max(target_matches)
        if not inclusive:
            output_end = output_end + 1
        return (output_start, output_end)

    def project_token_span(self, start, end, inclusive=False) -> Tuple[int, int]:
        """Project a span from source to target token sequence.

        Notes:
            When param inclusive=False, the end index is interpreted as exclusive,
            and the end of the span returned by the function will also be exclusive.
            When param inclusive=True, both start and end indexes are interpreted as inclusive,
            and the span returned by the function will also be inclusive.

        Examples:
            >>> source_tokens = ['abc', 'def', 'ghi', 'jkl']
            >>> target_tokens = ['abc', 'd', 'ef', 'ghi', 'jkl']
            >>> ta = TokenAligner(source_tokens, target_tokens)
            >>> start, end = 0, 2
            >>> print(ta.project_token_span(start, end))
            (0, 3)

        Raise:
            When target span is empty

        Returns:
            Tuple[int, int] representing the target span corresponding to the source span.
        """
        return self._project_span(
            mat=self.source_token_idx_to_target_token_idx, start=start, end=end, inclusive=inclusive
        )

    def project_token_to_char_span(self, start, end, inclusive=False) -> Tuple[int, int]:
        """Project a span from source to target token sequence.

        Notes:
            When param inclusive=False, the end index is interpreted as exclusive,
            and the end of the span returned by the function will also be exclusive.
            When param inclusive=True, both start and end indexes are interpreted as inclusive,
            and the span returned by the function will also be inclusive.

        Examples:
            >>> source_tokens = ['abc', 'def', 'ghi', 'jkl']
            >>> target_str = 'abc d ef ghi jkl'
            >>> ta = TokenAligner(source_tokens, target_str)
            >>> start, end = 0, 2
            >>> print(ta.project_token_to_char_span(start, end))
            (0, 8)

        Raise:
            When target span is empty

        Returns:
            Tuple[int, int] representing the target span corresponding to the source span.
        """
        return self._project_span(
            mat=self.source_token_idx_to_target_char_idx, start=start, end=end, inclusive=inclusive
        )

    def project_char_to_token_span(self, start, end, inclusive=False) -> Tuple[int, int]:
        """Project a span from source to target token sequence.

        Notes:
            When param inclusive=False, the end index is interpreted as exclusive,
            and the end of the span returned by the function will also be exclusive.
            When param inclusive=True, both start and end indexes are interpreted as inclusive,
            and the span returned by the function will also be inclusive.

        Examples:
            >>> source_str = 'abc def ghi jkl'
            >>> target_tokens = ['abc', 'd', 'ef', 'ghi', 'jkl']
            >>> ta = TokenAligner(source_str, target_tokens)
            >>> start, end = 0, 4
            >>> print(ta.project_char_to_token_span(start, end))
            (0, 1)

        Raise:
            When target span is empty

        Returns:
            Tuple[int, int] representing the target span corresponding to the source span.
        """
        return self._project_span(
            mat=self.source_char_idx_to_target_token_idx, start=start, end=end, inclusive=inclusive
        )
