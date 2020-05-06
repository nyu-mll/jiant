# -*- coding: utf-8 -*-
"""This module contains utilities for manipulating configs."""
from typing import List

import json
import _jsonnet  # type: ignore


def json_merge_patch(target_json: str, patch_json: str) -> str:
    """Merge json objects according to JSON merge patch spec: https://tools.ietf.org/html/rfc7396.

    Takes a target json string, and a patch json string and applies the patch json to the target
    json according to "JSON Merge Patch" (defined by https://tools.ietf.org/html/rfc7396).

    Args:
        target_json: the json to be overwritten by the patch json.
        patch_json: the json used to overwrite the target json.

    Returns:
        json str after applying the patch json to the target json using "JSON Merge Patch" method.

    """
    merged: str = """local target = {target_json};
                     local patch = {patch_json};
                     std.mergePatch(target, patch)""".format(
        target_json=target_json, patch_json=patch_json
    )
    return _jsonnet.evaluate_snippet("snippet", merged)


def merge_jsons_in_order(jsons: List[str]) -> str:
    """Applies JSON Merge Patch process to a list of json documents in order.

    Takes a list of json document strings and performs "JSON Merge Patch" (see json_merge_patch).
    The first element in the list of json docs is treated as the base, subsequent docs (if any)
    are applied as patches in order from first to last.

    Args:
        jsons: list of json docs to merge into a composite json document.

    Returns:
        The composite json document string.

    """
    base_json = jsons.pop(0)
    # json.loads is called to check that input strings are valid json.
    json.loads(base_json)
    composite_json = base_json
    for json_str in jsons:
        json.loads(json_str)
        composite_json = json_merge_patch(composite_json, json_str)
    return composite_json
