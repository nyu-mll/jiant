import pytest

import jiant.utils.python.functional as py_functional


def test_indexer():
    assert py_functional.indexer(1)({1: 2}) == 2
    with pytest.raises(KeyError):
        py_functional.indexer("1")({1: 2})
