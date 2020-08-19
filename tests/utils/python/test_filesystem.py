import os

import jiant.utils.python.filesystem as py_filesystem


def test_get_code_base_path():
    code_base_path = py_filesystem.get_code_base_path()
    assert os.path.exists(code_base_path)


def test_get_code_asset_path():
    import jiant

    assert py_filesystem.get_code_asset_path(os.path.join("jiant", "__init__.py")) == jiant.__file__
