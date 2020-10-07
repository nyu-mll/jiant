import numpy as np

from typing import NamedTuple, Sequence


class InclusiveSpan(NamedTuple):
    start: int
    end: int

    def to_slice(self):
        return slice(self.start, self.end + 1)

    def to_inclusive(self):
        return self

    def to_exclusive(self):
        return ExclusiveSpan(start=self.start, end=self.end + 1)


class ExclusiveSpan(NamedTuple):
    start: int
    end: int

    def to_slice(self):
        return slice(self.start, self.end)

    def to_inclusive(self):
        return ExclusiveSpan(start=self.start, end=self.end - 1)

    def to_exclusive(self):
        return self


def truncate_sequences(tokens_ls: Sequence[Sequence], max_length: int, truncate_end: bool = True):
    """Takes a sequence of sequences and trims the sub-seqs to fit within the max length.

    Trims the length of subsequences within a sequence until the total length of the combined sub-
    sequences is within the max_length. While total length exceeds the max_length, trims whichever
    subsequence is longest until the total combined length of the subsequences is under the limit.

    Args:
        tokens_ls (Sequence[Sequence]): sequence of subsequences to truncate.
        max_length (int): the maximum length of the combined sub-sequences.
        truncate_end (bool): if True, truncate from the end of the sub-sequence, else from start.

    Returns:
        Sequence[Sequence] with subsequences trimmed to fit within the max length when combined.

    """
    if len(tokens_ls) == 0:
        return []
    if len(tokens_ls) == 1:
        if truncate_end:
            return [tokens_ls[0][:max_length]]
        else:
            return [tokens_ls[0][-max_length:]]
    lengths = np.array([len(tokens) for tokens in tokens_ls])
    total_length = lengths.sum()
    if total_length < max_length:
        return tokens_ls
    target_lengths = lengths
    while sum(target_lengths) > max_length:
        target_lengths[np.argmax(target_lengths)] -= 1

    return [
        tokens[:target_length] if truncate_end else tokens[-target_length:]
        for tokens, target_length in zip(tokens_ls, target_lengths)
    ]


def pad_to_max_seq_length(ls, max_seq_length, pad_idx=0, pad_right=True, check=True):
    """Apply padding to an input sequence.

    Args:
        ls: sequence to pad.
        max_seq_length: max length up to which to apply padding.
        pad_idx: element to use for padding.
        pad_right: True if padding is applied to right side of sequence, False to pad on left side.
        check: True if result length should be checked as under the max sequence length.

    Returns:
        Sequence with specified padding applied.

    """
    padding = [pad_idx] * (max_seq_length - len(ls))
    if pad_right:
        result = ls + padding
    else:
        result = padding + ls

    if check:
        assert len(result) == max_seq_length
    return result
