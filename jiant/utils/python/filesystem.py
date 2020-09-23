import os
import importlib
import sys
from contextlib import contextmanager


def find_files(base_path, func):
    return sorted(
        [
            os.path.join(dp, filename)
            for dp, dn, filenames in os.walk(base_path)
            for filename in filenames
            if func(filename)
        ]
    )


def find_files_with_ext(base_path, ext):
    return find_files(base_path=base_path, func=lambda filename: filename.endswith(f".{ext}"))


def get_code_base_path():
    """Gets path to root of jiant code base

    Returns:
        Path to root of jiant code base
    """
    return os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, os.pardir,))


def get_code_asset_path(*rel_path):
    """Get path to file/folder within code base

    Like os.path.join, you can supple either arguments:
        "path", "to", "file"
     or
        "path/to/file"

    Args:
        *rel_path: one or more strings representing folder/file name,
                   similar to os.path.join(*rel_path)

    Returns:
        Path to file/folder within code base
    """
    return os.path.join(get_code_base_path(), *rel_path)


def find_case_insensitive_filename(filename, path):
    for f in os.listdir(path):
        if f.lower() == filename.lower():
            return f


@contextmanager
def temporarily_add_sys_path(path):
    sys.path = [path] + sys.path
    yield
    sys.path = sys.path[1:]


def import_from_path(path):
    base_path, filename = os.path.split(path)
    module_name = filename[:-3] if filename.endswith(".py") else filename
    with temporarily_add_sys_path(base_path):
        module = importlib.import_module(module_name)
    return module
