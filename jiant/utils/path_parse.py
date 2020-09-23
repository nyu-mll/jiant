import glob
import re


def tags_to_regex(tag_pattern, format_dict=None, default_format="\\w+"):
    r"""Converts a string with tags to regex.

    E.g. converts
        '/path/to/experiments/{model}/{task}'
    to
        '/path/to/experiments/(?P<model>\\w+)/(?P<task>\\w+)'


    Args:
        tag_pattern: str
            String with curly-brackets as tags
        format_dict: Dict[str, str]
            Special regex formatting for each tag
        default_format: str
            Default format if not format specified for tag in format_dict

    Returns: str
        regex string
    """
    if format_dict is None:
        format_dict = {}
    last_end = 0
    new_tokens = []
    for m in re.finditer("\\{(?P<tag>\\w+)\\}", tag_pattern):
        start, end = m.span()
        tag = m["tag"]
        new_tokens.append(tag_pattern[last_end:start])
        tag_format = format_dict.get(tag, default_format)
        new_tokens.append(f"(?P<{tag}>{tag_format})")
        last_end = end
    new_tokens.append(tag_pattern[last_end:])
    new_pattern = "".join(new_tokens)
    return new_pattern


def match_paths(path_pattern, format_dict=None, default_format="\\w+"):
    """Given a pattern, return a list of matches with tag values.

    E.g. converts
        '/path/to/experiments/{model}/{task}'
    to a list of dicts:
        {
            "model": ...,
            "task": ...,
            "path": ...,
        }

    Args:
        path_pattern: str
            Also used as input `tag_pattern` to tags_to_regex
        format_dict: Dict[str, str]
            See: tags_to_regex
        default_format: str
            See: tags_to_regex

    Returns: List[dict]
        List of matches
    """
    path_ls = sorted(glob.glob(re.sub(r"{(\w+)}", "*", path_pattern)))
    return match_path_ls(
        path_ls=path_ls,
        path_pattern=path_pattern,
        format_dict=format_dict,
        default_format=default_format,
    )


def match_path_ls(path_ls, path_pattern, format_dict=None, default_format="\\w+"):
    """Matches paths to each pattern

    Args:
        path_ls: List[str]
            List of paths
        path_pattern: str
            Input `tag_pattern` to tags_to_regex
        format_dict: Dict[str, str]
            See: tags_to_regex
        default_format: str
            See: tags_to_regex

    Returns: List[dict]
        List of matches
    """
    regex = re.compile(
        tags_to_regex(path_pattern, format_dict=format_dict, default_format=default_format,)
    )
    result_ls = []
    for path in path_ls:
        result = next(regex.finditer(path)).groupdict()
        assert "path" not in result, 'keyword clash: "path"'
        result["path"] = path
        result_ls.append(result)
    return result_ls
