import csv
import os
import shutil
import tempfile
import unittest
import pyhocon
import jsondiff

import jiant.utils.utils as utils


class TestParseJsonDiff(unittest.TestCase):
    def test_replace_insert_parse_json_diff(self):
        output_diff = {
            "mrpc": {
                jsondiff.replace: pyhocon.ConfigTree(
                    [
                        ("classifier_dropout", 0.1),
                        ("classifier_hid_dim", 256),
                        ("max_vals", 8),
                        ("val_interval", 1),
                    ]
                )
            }
        }
        parsed_diff = utils.parse_json_diff(output_diff)
        assert isinstance(parsed_diff["mrpc"], pyhocon.ConfigTree)
        output_diff = {
            "mrpc": {
                jsondiff.insert: pyhocon.ConfigTree(
                    [
                        ("classifier_dropout", 0.1),
                        ("classifier_hid_dim", 256),
                        ("max_vals", 8),
                        ("val_interval", 1),
                    ]
                )
            }
        }
        parsed_diff = utils.parse_json_diff(output_diff)
        assert isinstance(parsed_diff["mrpc"], pyhocon.ConfigTree)

    def test_delete_parse_json_diff(self):
        output_diff = {"mrpc": {jsondiff.delete: [1], "lr": 0.001}}
        parsed_diff = utils.parse_json_diff(output_diff)
        assert jsondiff.delete not in parsed_diff["mrpc"].keys()


class TestSortRecursively(unittest.TestCase):
    def test_replace_insert_parse_json_diff(self):
        input_diff = {
            "mrpc": pyhocon.ConfigTree(
                [
                    ("classifier_hid_dim", 256),
                    ("max_vals", 8),
                    ("classifier_dropout", 0.1),
                    ("val_interval", 1),
                ]
            ),
            "rte": {"configs": pyhocon.ConfigTree([("b", 1), ("a", 3)])},
        }
        sorted_diff = utils.sort_param_recursive(input_diff)
        assert list(sorted_diff["mrpc"].items()) == [
            ("classifier_dropout", 0.1),
            ("classifier_hid_dim", 256),
            ("max_vals", 8),
            ("val_interval", 1),
        ]
        assert list(sorted_diff["rte"]["configs"].items()) == [("a", 3), ("b", 1)]
