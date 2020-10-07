from typing import Any


def getter(attr_name: Any):
    def f(obj):
        return getattr(obj, attr_name)

    return f


def indexer(key):
    def f(obj):
        return obj[key]

    return f


def identity(*args):
    if len(args) > 1:
        return args
    else:
        return args[0]


# noinspection PyUnusedLocal
def always_false(*args, **kwargs):
    return False


# noinspection PyUnusedLocal
def always_true(*args, **kwargs):
    return True
