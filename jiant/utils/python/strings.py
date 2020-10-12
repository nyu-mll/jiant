def remove_prefix(s, prefix):
    assert s.startswith(prefix)
    return s[len(prefix) :]


def remove_suffix(s, suffix):
    assert s.endswith(suffix)
    return s[: -len(suffix)]


def replace_prefix(s, prefix, new_prefix):
    return new_prefix + remove_prefix(s, prefix)


def replace_suffix(s, suffix, new_suffix):
    return remove_suffix(s, suffix) + new_suffix
