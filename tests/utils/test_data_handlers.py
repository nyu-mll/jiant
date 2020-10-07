# -*- coding: utf-8 -*-
import os

from jiant.utils import data_handlers


def test_md5_checksum_matches_expected_checksum():
    expected_md5_checksum = "4d5e587120171bc1ba4d49e2aa862a12"  # calc'd w/ http://onlinemd5.com/
    filepath = os.path.join(os.path.dirname(__file__), "config/base_config.json")
    computed_md5_checksum = data_handlers.md5_checksum(filepath)
    assert expected_md5_checksum == computed_md5_checksum
