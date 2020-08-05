import numpy as np
import pytest

from jiant.utils.retokenize import TokenAligner, token_to_char


def test_token_to_char():
    expected_m = np.array(
        # T  h  i  s     i  s     a     t  e  s  t
        [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # This
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # is
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # a
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        ],  # test
        dtype=np.int32,
    )
    m = token_to_char("This is a test")
    assert (m == expected_m).all()


def test_token_to_char_no_spaces():
    expected_m = np.array(
        # T  h  i  s  i  s  a  t  e  s  t
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],  # Thisisatest
        dtype=np.int32,
    )
    m = token_to_char("Thisisatest")
    assert (m == expected_m).all()


def test_token_to_char_empty_str():
    expected_m = np.array([[]], dtype=np.int32)
    m = token_to_char("")
    assert (m == expected_m).all()


def test_token_aligner_project_single_token_index():
    source_tokens = ["abc", "def", "ghi", "jkl"]
    target_tokens = ["abc", "d", "ef", "ghi", "jkl"]
    ta = TokenAligner(source_tokens, target_tokens)
    m = ta.project_token_idxs(1)
    m_expected = np.array([1, 2])
    assert (m == m_expected).all()


def test_token_aligner_project_multiple_token_indices():
    source_tokens = ["abc", "def", "ghi", "jkl"]
    target_tokens = ["abc", "d", "ef", "ghi", "jkl"]
    ta = TokenAligner(source_tokens, target_tokens)
    m = ta.project_token_idxs([1, 3])
    m_expected = np.array([1, 2, 4])
    assert (m == m_expected).all()


def test_token_aligner_project_to_empty_target_token_sequence():
    source_tokens = ["abc", "def", "ghi", "jkl"]
    target_tokens = []
    ta = TokenAligner(source_tokens, target_tokens)
    m = ta.project_token_idxs([1, 3])
    m_expected = np.array([])
    assert (m == m_expected).all()


def test_token_aligner_project_to_mismatched_token_sequence():
    source_tokens = ["abc", "def", "ghi", "jkl"]
    target_tokens = ["qrs", "tuv", "wxy", "z"]
    ta = TokenAligner(source_tokens, target_tokens)
    m = ta.project_token_idxs([1])
    m_expected = np.array([])
    assert (m == m_expected).all()


def test_token_aligner_project_token_span():
    source_tokens = ["abc", "def", "ghi", "jkl"]
    target_tokens = ["abc", "d", "ef", "ghi", "jkl"]
    ta = TokenAligner(source_tokens, target_tokens)
    m = ta.project_token_span(1, 2)
    m_expected = np.array([1, 3])
    assert (m == m_expected).all()


def test_token_aligner_project_token_span_last_token_range_is_end_exclusive():
    source_tokens = ["abc", "def", "ghi", "jkl"]
    target_tokens = ["abc", "d", "ef", "ghi", "jkl"]
    ta = TokenAligner(source_tokens, target_tokens)
    m = ta.project_token_span(3, 4)
    m_expected = np.array([4, 5])
    assert (m == m_expected).all()


def test_wpm_tok_idx_proj_1():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_token_index = [[0], [1], [2], [3], [4], [5], [6]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_wpm_tok_idx_proj_2():
    src_tokens = ["I", "look", "at", "Sarah's", "dog.", "It", "was", "cute.!"]
    tgt_tokens = ["I", "look", "at", "Sarah", "'", "s", "dog", ".", "It", "was", "cute", ".", "!"]
    tgt_token_index = [[0], [1], [2], [3, 4, 5], [6, 7], [8], [9], [10, 11, 12]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_wpm_tok_idx_proj_3():
    src_tokens = [
        "Mr.",
        "Immelt",
        "chose",
        "to",
        "focus",
        "on",
        "the",
        "incomprehensibility",
        "of",
        "accounting",
        "rules.",
    ]
    tgt_tokens = [
        "Mr",
        ".",
        "I",
        "##mme",
        "##lt",
        "chose",
        "to",
        "focus",
        "on",
        "the",
        "in",
        "##com",
        "##p",
        "##re",
        "##hen",
        "##si",
        "##bility",
        "of",
        "accounting",
        "rules",
        ".",
    ]
    tgt_token_index = [
        [0, 1],
        [2, 3, 4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10, 11, 12, 13, 14, 15, 16],
        [17],
        [18],
        [19, 20],
    ]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_wpm_tok_idx_proj_4():
    src_tokens = ["What?"]
    tgt_tokens = ["What", "?"]
    tgt_token_index = [[0, 1]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_moses_tok_idx_proj_1():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_token_index = [[0], [1], [2], [3], [4], [5], [6]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_moses_tok_idx_proj_2():
    src_tokens = ["I", "look", "at", "Sarah's", "dog.", "It", "was", "cute.!"]
    tgt_tokens = ["I", "look", "at", "Sarah", "&apos;s", "dog", ".", "It", "was", "cute", ".", "!"]
    tgt_token_index = [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10, 11]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_moses_tok_idx_proj_3():
    src_tokens = [
        "Mr.",
        "Immelt",
        "chose",
        "to",
        "focus",
        "on",
        "the",
        "incomprehensibility",
        "of",
        "accounting",
        "rules.",
    ]
    tgt_tokens = [
        "Mr.",
        "Immelt",
        "chose",
        "to",
        "focus",
        "on",
        "the",
        "incomprehensibility",
        "of",
        "accounting",
        "rules",
        ".",
    ]
    tgt_token_index = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10, 11]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_moses_tok_idx_proj_4():
    src_tokens = ["What?"]
    tgt_tokens = ["What", "?"]
    tgt_token_index = [[0, 1]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_bpe_tok_idx_proj_1():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = [
        "members</w>",
        "of</w>",
        "the</w>",
        "house</w>",
        "clapped</w>",
        "their</w>",
        "hands</w>",
    ]
    tgt_token_index = [[0], [1], [2], [3], [4], [5], [6]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_bpe_tok_idx_proj_2():
    src_tokens = ["I", "look", "at", "Sarah's", "dog.", "It", "was", "cute.!"]
    tgt_tokens = [
        "i</w>",
        "look</w>",
        "at</w>",
        "sarah</w>",
        "'s</w>",
        "dog</w>",
        ".</w>",
        "it</w>",
        "was</w>",
        "cute</w>",
        ".</w>",
        "!</w>",
    ]
    tgt_token_index = [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10, 11]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_bpe_tok_idx_proj_3():
    src_tokens = [
        "Mr.",
        "Immelt",
        "chose",
        "to",
        "focus",
        "on",
        "the",
        "incomprehensibility",
        "of",
        "accounting",
        "rules.",
    ]
    tgt_tokens = [
        "mr.</w>",
        "im",
        "melt</w>",
        "chose</w>",
        "to</w>",
        "focus</w>",
        "on</w>",
        "the</w>",
        "in",
        "comprehen",
        "si",
        "bility</w>",
        "of</w>",
        "accounting</w>",
        "rules</w>",
        ".</w>",
    ]
    tgt_token_index = [[0], [1, 2], [3], [4], [5], [6], [7], [8, 9, 10, 11], [12], [13], [14, 15]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_bpe_tok_idx_proj_4():
    src_tokens = ["What?"]
    tgt_tokens = ["what</w>", "?</w>"]
    tgt_token_index = [[0, 1]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_sentencepiece_tok_idx_proj_1():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = ["▁Members", "▁of", "▁the", "▁House", "▁clapped", "▁their", "▁hands"]
    tgt_token_index = [[0], [1], [2], [3], [4], [5], [6]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_sentencepiece_tok_idx_proj_2():
    src_tokens = ["I", "look", "at", "Sarah's", "dog.", "It", "was", "cute.!"]
    tgt_tokens = [
        "▁I",
        "▁look",
        "▁at",
        "▁Sarah",
        "'",
        "s",
        "▁dog",
        ".",
        "▁It",
        "▁was",
        "▁cute",
        ".",
        "!",
    ]
    tgt_token_index = [[0], [1], [2], [3, 4, 5], [6, 7], [8], [9], [10, 11, 12]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_sentencepiece_tok_idx_proj_3():
    src_tokens = [
        "Mr.",
        "Immelt",
        "chose",
        "to",
        "focus",
        "on",
        "the",
        "incomprehensibility",
        "of",
        "accounting",
        "rules.",
    ]
    tgt_tokens = [
        "▁Mr",
        ".",
        "▁I",
        "m",
        "mel",
        "t",
        "▁chose",
        "▁to",
        "▁focus",
        "▁on",
        "▁the",
        "▁in",
        "comp",
        "re",
        "hen",
        "s",
        "ibility",
        "▁of",
        "▁accounting",
        "▁rules",
        ".",
    ]
    tgt_token_index = [
        [0, 1],
        [2, 3, 4, 5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [11, 12, 13, 14, 15, 16],
        [17],
        [18],
        [19, 20],
    ]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_sentencepiece_tok_idx_proj_4():
    src_tokens = ["What?"]
    tgt_tokens = ["▁What", "?"]
    tgt_token_index = [[0, 1]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_bytebpe_tok_idx_proj_1():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = ["Members", "Ġof", "Ġthe", "ĠHouse", "Ġcl", "apped", "Ġtheir", "Ġhands"]
    tgt_token_index = [[0], [1], [2], [3], [4, 5], [6], [7]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_bytebpe_tok_idx_proj_2():
    src_tokens = ["I", "look", "at", "Sarah's", "dog.", "It", "was", "cute.!"]
    tgt_tokens = [
        "I",
        "Ġlook",
        "Ġat",
        "ĠSarah",
        "'s",
        "Ġdog",
        ".",
        "ĠIt",
        "Ġwas",
        "Ġcute",
        ".",
        "!",
    ]
    tgt_token_index = [[0], [1], [2], [3, 4], [5, 6], [7], [8], [9, 10, 11]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_bytebpe_tok_idx_proj_3():
    src_tokens = [
        "Mr.",
        "Immelt",
        "chose",
        "to",
        "focus",
        "on",
        "the",
        "incomprehensibility",
        "of",
        "accounting",
        "rules.",
    ]
    tgt_tokens = [
        "Mr",
        ".",
        "ĠImm",
        "elt",
        "Ġchose",
        "Ġto",
        "Ġfocus",
        "Ġon",
        "Ġthe",
        "Ġincomp",
        "rehens",
        "ibility",
        "Ġof",
        "Ġaccounting",
        "Ġrules",
        ".",
    ]
    tgt_token_index = [[0, 1], [2, 3], [4], [5], [6], [7], [8], [9, 10, 11], [12], [13], [14, 15]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


def test_bytebpe_tok_idx_proj_4():
    src_tokens = ["What?"]
    tgt_tokens = ["What", "?"]
    tgt_token_index = [[0, 1]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    for src_token_idx in range(len(src_tokens)):
        projected_tgt_tok_idx = ta.project_token_idxs(src_token_idx)
        assert (tgt_token_index[src_token_idx] == projected_tgt_tok_idx).all()


"""
Note: test_project_token_span_* tests use same data and token aligner as test_bytebpe_tok_idx_proj_1. 
Token projection is tested extensively above, the focus of the following tests is span projection. 
For span prediction token token projection does the heavy lifting — tests covering span prediction 
need to check that careful indexing.
"""


def test_project_token_span_single_token():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = ["Members", "Ġof", "Ġthe", "ĠHouse", "Ġcl", "apped", "Ġtheir", "Ġhands"]
    # reference: tgt_token_index = [[0], [1], [2], [3], [4, 5], [6], [7]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    assert (0, 1) == ta.project_token_span(0, 1)


def test_project_token_span_covering_multiple_source_sequence_tokens():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = ["Members", "Ġof", "Ġthe", "ĠHouse", "Ġcl", "apped", "Ġtheir", "Ġhands"]
    # reference: tgt_token_index = [[0], [1], [2], [3], [4, 5], [6], [7]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    assert (0, 2) == ta.project_token_span(0, 2)


def test_project_token_span_covering_multiple_target_sequence_tokens():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = ["Members", "Ġof", "Ġthe", "ĠHouse", "Ġcl", "apped", "Ġtheir", "Ġhands"]
    # reference: tgt_token_index = [[0], [1], [2], [3], [4, 5], [6], [7]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    assert (4, 6) == ta.project_token_span(4, 5)


def test_project_token_span_covering_whole_sequence():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = ["Members", "Ġof", "Ġthe", "ĠHouse", "Ġcl", "apped", "Ġtheir", "Ġhands"]
    # reference: tgt_token_index = [[0], [1], [2], [3], [4, 5], [6], [7]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    assert (0, 8) == ta.project_token_span(0, 7)


def test_project_invalid_span():
    src_tokens = ["Members", "of", "the", "House", "clapped", "their", "hands"]
    tgt_tokens = ["Members", "Ġof", "Ġthe", "ĠHouse", "Ġcl", "apped", "Ġtheir", "Ġhands"]
    # reference: tgt_token_index = [[0], [1], [2], [3], [4, 5], [6], [7]]
    ta = TokenAligner(src_tokens, tgt_tokens)
    with pytest.raises(ValueError):
        ta.project_token_span(0, 0)


def test_private_project_token_span():
    mat = np.eye(5, dtype=int)
    mat[0][0] = 0
    mat[3][3] = 0
    assert TokenAligner._project_span(mat, 1, 3, inclusive=True) == (1, 2)
    assert TokenAligner._project_span(mat, 1, 3, inclusive=False) == (1, 3)
    assert TokenAligner._project_span(mat, 1, 2, inclusive=True) == (1, 2)
    assert TokenAligner._project_span(mat, 1, 2, inclusive=False) == (1, 2)
    assert TokenAligner._project_span(mat, 1, 4, inclusive=True) == (1, 4)
    assert TokenAligner._project_span(mat, 1, 4, inclusive=False) == (1, 3)
