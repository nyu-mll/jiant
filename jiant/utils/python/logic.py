from typing import Any, Optional


def replace_none(elem: Optional[Any], default: Any):
    """If elem is None, return default, else return elem

    Args:
        elem: element to possibly return
        default: default element

    Returns:
        elem, or default

    """
    if elem is None:
        return default
    else:
        return elem
