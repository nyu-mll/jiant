import os
import glob
import json


def read_file(path, mode="r", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        return f.read()


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def read_json(path, mode="r", **kwargs):
    return json.loads(read_file(path, mode=mode, **kwargs))


def write_json(data, path):
    return write_file(json.dumps(data, indent=2), path)


def read_jsonl(path, mode="r", **kwargs):
    # Manually open because .splitlines is different from iterating over lines
    ls = []
    with open(path, mode, **kwargs) as f:
        for line in f:
            ls.append(json.loads(line))
    return ls


def write_jsonl(data, path):
    assert isinstance(data, list)
    lines = [to_jsonl(elem) for elem in data]
    write_file("\n".join(lines), path)


def read_file_lines(path, mode="r", encoding="utf-8", strip_lines=False, **kwargs):
    with open(path, mode=mode, encoding=encoding, **kwargs) as f:
        lines = f.readlines()
    if strip_lines:
        return [line.strip() for line in lines]
    else:
        return lines


def read_json_lines(path, mode="r", encoding="utf-8", **kwargs):
    with open(path, mode=mode, encoding=encoding, **kwargs) as f:
        for line in f.readlines():
            yield json.loads(line)


def to_jsonl(data):
    return json.dumps(data).replace("\n", "")


def create_containing_folder(path):
    fol_path = os.path.split(path)[0]
    os.makedirs(fol_path, exist_ok=True)


def sorted_glob(pathname, *, recursive=False):
    return sorted(glob.glob(pathname, recursive=recursive))


def assert_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def assert_not_exists(path):
    if os.path.exists(path):
        raise FileExistsError(path)


def get_num_lines(path):
    with open(path, "r") as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def create_dir(*args):
    """Makes a folder and returns the path

    Args:
        *args: args to os.path.join
    """
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path
