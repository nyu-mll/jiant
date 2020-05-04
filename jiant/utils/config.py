# -*- coding: utf-8 -*-
"""This module contains utilities for manipulating configs."""
import _jsonnet  # type: ignore


def json_merge_patch(target_json: str, patch_json: str) -> str:
    """Merge json objects according to JSON merge patch spec: https://tools.ietf.org/html/rfc7396

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
