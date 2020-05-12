import jiant.utils.python.checks as py_checks


def test_dict_equal():
    assert py_checks.dict_equal({1: 2}, {1: 2})
    assert not py_checks.dict_equal({1: 2}, {1: 3})
    assert not py_checks.dict_equal({1: 2}, {2: 2})
    assert not py_checks.dict_equal({1: 2}, {2: 2, 1: 1})
