# -*- coding: utf-8 -*-
import json
from jiant.utils import config


def test_json_merge_patch():
    """Tests that JSON Merge Patch works as expected (https://tools.ietf.org/html/rfc7396)"""
    target = """
    {
        "title": "Goodbye!",
        "author" : {
            "givenName" : "John",
            "familyName" : "Doe"
        },
        "tags":[ "example", "sample" ],
        "content": "This will be unchanged"
    }
    """
    patch = """
    {
        "title": "Hello!",
        "phoneNumber": "+01-123-456-7890",
        "author": {
            "familyName": null
        },
        "tags": [ "example" ]
    }
    """
    merged = config.json_merge_patch(target, patch)
    expected = """
    {
        "title": "Hello!",
        "author" : {
            "givenName" : "John"
        },
        "tags": [ "example" ],
        "content": "This will be unchanged",
        "phoneNumber": "+01-123-456-7890"
    }
    """
    merged_sorted: str = json.dumps(json.loads(merged), sort_keys=True)
    expected_sorted: str = json.dumps(json.loads(expected), sort_keys=True)
    assert merged_sorted == expected_sorted
