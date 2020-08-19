import json
import pytest
from dataclasses import dataclass

import jiant.utils.python.datastructures as py_datastructures


def test_take_one():
    assert py_datastructures.take_one([9]) == 9
    assert py_datastructures.take_one((9,)) == 9
    assert py_datastructures.take_one({9}) == 9
    assert py_datastructures.take_one("9") == "9"
    assert py_datastructures.take_one({9: 10}) == 9

    with pytest.raises(IndexError):
        py_datastructures.take_one([])
    with pytest.raises(IndexError):
        py_datastructures.take_one([1, 2])
    with pytest.raises(IndexError):
        py_datastructures.take_one("2342134")


def test_chain_idx():
    # dict
    d = {1: {2: 3}}
    ls = [1, [2], [None, [3]]]
    assert py_datastructures.chain_idx(d[1], [2]) == 3
    assert py_datastructures.chain_idx(d, [1, 2]) == 3
    with pytest.raises(KeyError):
        py_datastructures.chain_idx(d, [1, 3])
    with pytest.raises(KeyError):
        py_datastructures.chain_idx(d, [3])

    # list
    assert py_datastructures.chain_idx(ls, [0]) == 1
    assert py_datastructures.chain_idx(ls, [1]) == [2]
    assert py_datastructures.chain_idx(ls, [1, 0]) == 2
    assert py_datastructures.chain_idx(ls, [2, 0]) is None
    assert py_datastructures.chain_idx(ls, [2, 1, 0]) == 3
    with pytest.raises(TypeError):
        py_datastructures.chain_idx(ls, [0, 0, 0])
    with pytest.raises(IndexError):
        py_datastructures.chain_idx(ls, [1, 1])


def test_chain_idx_get():
    # dict
    d = {1: {2: 3}}
    ls = [1, [2], [None, [3]]]
    assert py_datastructures.chain_idx_get(d[1], [2], default=1234) == 3
    assert py_datastructures.chain_idx_get(d, [1, 3], default=1234) == 1234

    # list
    assert py_datastructures.chain_idx_get(ls, [0], default=1234) == 1
    assert py_datastructures.chain_idx_get(ls, [0, 0], default=1234) == 1234
    assert py_datastructures.chain_idx_get(ls, [1, 1], default=1234) == 1234


def test_combine_dicts_with_disjoint_key_sets():
    combined_dict = py_datastructures.combine_dicts([{"d1_k1": "d1_v1"}, {"d2_k1": "d2_v1"}])
    expected_dict = {"d1_k1": "d1_v1", "d2_k1": "d2_v1"}
    combined_sorted: str = json.dumps(combined_dict, sort_keys=True)
    expected_sorted: str = json.dumps(expected_dict, sort_keys=True)
    assert combined_sorted == expected_sorted


def test_combine_dicts_with_overlapping_key_sets():
    with pytest.raises(RuntimeError):
        py_datastructures.combine_dicts([{"k1": "v1"}, {"k1": "v1"}])


def test_partition_list():
    assert py_datastructures.partition_list(list(range(10)), 5) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9],
    ]
    assert py_datastructures.partition_list(list(range(10)), 3) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]
    assert py_datastructures.partition_list(list(range(10)), 1) == [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]


def test_extended_dataclass_mixin():
    @dataclass
    class MyClass(py_datastructures.ExtendedDataClassMixin):
        int1: int
        str1: str

    assert MyClass.get_fields() == ["int1", "str1"]


def test_check_keys():
    def create_dict_with_keys(keys):
        return {k: None for k in keys}

    # equal
    d1 = create_dict_with_keys([1, 2, 3])
    assert py_datastructures.check_keys(d1, [1, 2, 3])
    assert not py_datastructures.check_keys(d1, [1, 2, 3, 4])
    assert not py_datastructures.check_keys(d1, [1])
    assert not py_datastructures.check_keys(d1, [])

    # subset
    d1 = create_dict_with_keys([1, 2, 3])
    assert py_datastructures.check_keys(d1, [1, 2, 3], mode="subset")
    assert py_datastructures.check_keys(d1, [1, 2, 3, 4], mode="subset")
    assert not py_datastructures.check_keys(d1, [1, 2, 4], mode="subset")
    assert not py_datastructures.check_keys(d1, [1, 2], mode="subset")

    # superset
    d1 = create_dict_with_keys([1, 2, 3])
    assert py_datastructures.check_keys(d1, [1, 2, 3], mode="superset")
    assert not py_datastructures.check_keys(d1, [1, 2, 3, 4], mode="superset")
    assert not py_datastructures.check_keys(d1, [1, 2, 4], mode="superset")
    assert py_datastructures.check_keys(d1, [1, 2], mode="superset")

    # strict_subset
    d1 = create_dict_with_keys([1, 2, 3])
    assert not py_datastructures.check_keys(d1, [1, 2, 3], mode="strict_subset")
    assert py_datastructures.check_keys(d1, [1, 2, 3, 4], mode="strict_subset")
    assert not py_datastructures.check_keys(d1, [1, 2, 4], mode="strict_subset")
    assert not py_datastructures.check_keys(d1, [1, 2], mode="strict_subset")

    # strict_superset
    d1 = create_dict_with_keys([1, 2, 3])
    assert not py_datastructures.check_keys(d1, [1, 2, 3], mode="strict_superset")
    assert not py_datastructures.check_keys(d1, [1, 2, 3, 4], mode="strict_superset")
    assert not py_datastructures.check_keys(d1, [1, 2, 4], mode="strict_superset")
    assert py_datastructures.check_keys(d1, [1, 2], mode="strict_superset")

    with pytest.raises(AssertionError):
        py_datastructures.check_keys(d1, [1, 1])


def test_get_unique_list_in_order():
    assert py_datastructures.get_unique_list_in_order([[1, 2], [3], [4]]) == [1, 2, 3, 4]
    assert py_datastructures.get_unique_list_in_order([[1, 2, 3], [3], [4]]) == [1, 2, 3, 4]
    assert py_datastructures.get_unique_list_in_order([[1, 2, 3], [4], [3]]) == [1, 2, 3, 4]


def test_reorder_keys():
    dict1 = {"a": 1, "b": 1, "c": 1}
    assert list(py_datastructures.reorder_keys(dict1, key_list=["a", "b", "c"]).keys()) == [
        "a",
        "b",
        "c",
    ]
    assert list(py_datastructures.reorder_keys(dict1, key_list=["c", "a", "b"]).keys()) == [
        "c",
        "a",
        "b",
    ]
    with pytest.raises(AssertionError):
        py_datastructures.reorder_keys(dict1, key_list=["d", "b", "d"])


def test_set_dict_keys():
    d = {"a": 1, "b": 2, "c": 3}
    assert list(py_datastructures.set_dict_keys(d, ["a", "b", "c"])) == ["a", "b", "c"]
    assert list(py_datastructures.set_dict_keys(d, ["a", "c", "b"])) == ["a", "c", "b"]
    with pytest.raises(AssertionError):
        py_datastructures.set_dict_keys(d, ["a", "b"])
