import jiant.utils.python.logic as py_logic


def test_replace_none():
    assert py_logic.replace_none(1, default=2) == 1
    assert py_logic.replace_none(None, default=2) == 2
