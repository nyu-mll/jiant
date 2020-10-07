from jiant.tasks.utils import truncate_sequences


def test_truncate_empty_sequence():
    seq = []
    trunc_seq = truncate_sequences(seq, 10)
    assert not trunc_seq


def test_truncate_single_sequence_default_trunc_end():
    seq = [["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"]]
    trunc_seq = truncate_sequences(seq, 8)
    assert trunc_seq == [["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx"]]


def test_truncate_single_sequence_trunc_start():
    seq = [["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"]]
    trunc_seq = truncate_sequences(seq, 8, False)
    assert trunc_seq == [["def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"]]


def test_truncate_two_sequences_default_trunc_end():
    seqs = [["abc", "def", "ghi", "jkl"], ["mno", "pqr", "stu", "vwx", "yz"]]
    trunc_seqs = truncate_sequences(seqs, 8)
    assert trunc_seqs == [["abc", "def", "ghi", "jkl"], ["mno", "pqr", "stu", "vwx"]]


def test_truncate_more_than_two_sequences_trunc_start():
    seqs = [["abc", "def", "ghi"], ["jkl", "mno", "pqr"], ["stu", "vwx", "yz"]]
    trunc_seqs = truncate_sequences(seqs, 8)
    assert trunc_seqs == [["abc", "def"], ["jkl", "mno", "pqr"], ["stu", "vwx", "yz"]]


def test_truncate_two_sequences_default_trunc_start():
    seqs = [["abc", "def", "ghi", "jkl"], ["mno", "pqr", "stu", "vwx", "yz"]]
    trunc_seqs = truncate_sequences(seqs, 8, False)
    assert trunc_seqs == [["abc", "def", "ghi", "jkl"], ["pqr", "stu", "vwx", "yz"]]
