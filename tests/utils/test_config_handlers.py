# -*- coding: utf-8 -*-
import json
import os

from jiant.utils import config_handlers


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
    merged = config_handlers.json_merge_patch(target, patch)
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


def test_merging_multiple_json_configs():
    with open(os.path.join(os.path.dirname(__file__), "config/base_config.json")) as f:
        base_config = f.read()
    with open(os.path.join(os.path.dirname(__file__), "./config/first_override_config.json")) as f:
        override_config_1 = f.read()
    with open(os.path.join(os.path.dirname(__file__), "./config/second_override_config.json")) as f:
        override_config_2 = f.read()
    merged_config = config_handlers.merge_jsons_in_order(
        [base_config, override_config_1, override_config_2]
    )
    with open(os.path.join(os.path.dirname(__file__), "./config/final_config.json")) as f:
        expected_config = f.read()
    sorted_merged_config = json.dumps(json.loads(merged_config), sort_keys=True)
    sorted_expected_config = json.dumps(json.loads(expected_config), sort_keys=True)
    assert sorted_merged_config == sorted_expected_config
